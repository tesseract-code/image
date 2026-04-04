#version 410 core

// ---------------------------------------------------------------------------
// image.frag
//
// Processing pipeline (refactored into logical function blocks):
//
//   1. Texture sample
//   2. Normalize range
//   3. LUT / data transform  log | sqrt | square  (on scalar intensity)
//   4. Colormap              intensity → RGB via LUT texture
//   5. Contrast & brightness (linear light, before gamma)
//   6. Gamma correction
//   7. Color balance
//   8. Invert
//   9. Output clamp & alpha
//
// BGR / BGRA channel reordering is handled entirely via OpenGL texture swizzle
// masks set at upload time — this shader always receives RGB[A] order.
// ---------------------------------------------------------------------------

in  vec2 fragTexCoord;
out vec4 fragColor;

// Texture inputs
uniform sampler2D imageTexture;     // RGB / RGBA / Mono  (BGR reorder done on CPU via swizzle)
uniform sampler2D colormapTexture;  // 1-D colormap LUT (sampled along x, y = 0.5)

uniform bool use_cmap;

// --- Normalization ---
uniform float norm_vmin;
uniform float norm_vmax;

// --- Image enhancements ---
uniform float brightness;
uniform float contrast;
uniform float inv_gamma;
uniform vec3  color_balance;
uniform bool  invert;

// --- LUT / transfer function ---
// Applied to scalar intensity BEFORE colormap lookup.
// apply_lut() always receives input already normalized to [0, 1] by
// normalize_range(); no secondary remapping is performed inside the LUT.
// lut_type:  0 = Linear (pass-through)
//            1 = Logarithmic
//            2 = Square-root
//            3 = Square
uniform bool  lut_enabled;
uniform int   lut_type;


// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const vec3  BT709_LUMINANCE = vec3(0.2126, 0.7152, 0.0722);
const float EPSILON         = 1e-5;
const float GAMMA_THRESHOLD = 0.001;
// log(1 + 1) == log(2), so dividing by LOG2 maps log(x + 1) exactly onto
// [0, 1] for x in [0, 1].  Named constant avoids a per-fragment recompute.
const float LOG2            = log(2.0);


// ---------------------------------------------------------------------------
// Transfer Functions — all map [0, 1] → [0, 1]
// ---------------------------------------------------------------------------

// Log: compress bright end.
// log(x + 1.0) avoids the singularity at 0; dividing by LOG2 maps [0,1] → [0,1].
float lut_log(float x) {
    return log(x + 1.0) / LOG2;
}

// Sqrt: mild compression.
float lut_sqrt(float x) {
    return sqrt(x);
}

// Square: expand bright end / suppress low values.
float lut_square(float x) {
    return x * x;
}

// Dispatch helper — apply the selected transfer function.
float apply_lut(float x) {
    // Remap to [0, 1] using the dataset's own min/max before the transform.
    x = clamp((x - lut_min) * lut_norm_factor, 0.0, 1.0);

    if      (lut_type == 1) x = lut_log(x);
    else if (lut_type == 2) x = lut_sqrt(x);
    else if (lut_type == 3) x = lut_square(x);
    // lut_type == 0: linear, x unchanged.

    return x;
}


// ---------------------------------------------------------------------------
// Processing Pipeline Functions
// ---------------------------------------------------------------------------

vec4 sample_texture() {
    return texture(imageTexture, fragTexCoord);
}

// Normalize RGB to [0, 1] using the display range.
vec3 normalize_range(vec3 rgb) {
    float range = norm_vmax - norm_vmin;
    if (abs(range) > EPSILON) {
        rgb = (rgb - vec3(norm_vmin)) / range;
    }
    return rgb;
}

// Extract luminance for intensity-based processing.
float extract_luminance(vec3 rgb) {
    // For monochrome inputs R == G == B.  The BT.709 coefficients sum to
    // exactly 1.0, so the dot product correctly reduces to rgb.r.
    // For RGB inputs this yields a perceptually uniform intensity scalar
    // suitable for driving the data transform.
    return dot(rgb, BT709_LUMINANCE);
}

// Clamp to [0, 1] then apply the selected LUT transform.
float apply_intensity_transform(float intensity) {
    // Clamp before the transform: upstream normalisation can produce values
    // marginally outside [0, 1] due to floating-point rounding.
    intensity = clamp(intensity, 0.0, 1.0);
    if (lut_enabled) {
        intensity = apply_lut(intensity);
    }
    return intensity;
}

// Apply colormap or reconstruct RGB from the transformed intensity.
vec3 apply_colormap(vec3 rgb, float intensity) {
    if (use_cmap) {
        // Replace rgb entirely via colormap lookup.
        vec4 lutColor = texture(colormapTexture, vec2(intensity, 0.5));
        return lutColor.rgb;
    } else if (lut_enabled) {
        // No colormap, but a data transform was requested — scale RGB channels
        // proportionally to preserve hue while reflecting the new intensity.
        // Clamp the result: if the transform raises intensity above the original
        // luminance, individual channels can exceed 1.0.
        float lum = dot(rgb, BT709_LUMINANCE);
        vec3 result = (lum > EPSILON) ? rgb * (intensity / lum) : vec3(intensity);
        return clamp(result, 0.0, 1.0);
    }
    return rgb;
}

// Apply contrast and brightness adjustments.
// Operates in linear light, before gamma, so equal numeric steps are
// perceptually uniform and the two controls compose predictably.
vec3 apply_contrast_brightness(vec3 rgb) {
    // Clamp inputs first — upstream ops can push values outside [0, 1]
    // and unbounded values interact badly with extreme contrast settings.
    rgb = clamp(rgb, 0.0, 1.0);
    rgb = (rgb - 0.5) * contrast + 0.5 + brightness;
    return rgb;
}

// Apply gamma correction.
vec3 apply_gamma(vec3 rgb) {
    // Guard against negative base — pow() is undefined in GLSL for negative
    // bases with non-integer exponents.
    if (abs(inv_gamma - 1.0) > GAMMA_THRESHOLD) {
        rgb = pow(max(rgb, vec3(0.0)), vec3(inv_gamma));
    }
    return rgb;
}

// Apply per-channel color balance multipliers.
vec3 apply_color_balance(vec3 rgb) {
    return rgb * color_balance;
}

// Apply inversion.
vec3 apply_invert(vec3 rgb) {
    if (invert) {
        rgb = 1.0 - rgb;
    }
    return rgb;
}

// Clamp and pack final output.
vec4 finalize_output(vec3 rgb, float alpha) {
    return vec4(clamp(rgb, 0.0, 1.0), alpha);
}

// Returns true only when every pipeline stage is a no-op, including the
// normalization step (norm_vmin == 0, norm_vmax == 1).
// Skipping all processing lets the driver return the raw texel immediately.
bool should_skip_processing() {
    return !use_cmap
        && !lut_enabled
        && abs(norm_vmin) < EPSILON
        && abs(norm_vmax - 1.0) < EPSILON
        && abs(brightness) < EPSILON
        && abs(contrast - 1.0) < EPSILON
        && abs(inv_gamma - 1.0) < GAMMA_THRESHOLD
        && all(lessThan(abs(color_balance - vec3(1.0)), vec3(EPSILON)))
        && !invert;
}


// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
void main() {
    vec4 texColor = sample_texture();
    vec3 rgb = texColor.rgb;
    float alpha = texColor.a;

    if (should_skip_processing()) {
        fragColor = texColor;
        return;
    }

    rgb = normalize_range(rgb);

    // LUT / Colormap (in linear light).
    if (use_cmap || lut_enabled) {
        float intensity = extract_luminance(rgb);
        intensity = apply_intensity_transform(intensity);
        rgb = apply_colormap(rgb, intensity);
    }

    // Contrast & brightness must come before gamma so they operate in
    // linear light; gamma encodes for display last.
    rgb = apply_contrast_brightness(rgb);
    rgb = apply_gamma(rgb);
    rgb = apply_color_balance(rgb);
    rgb = apply_invert(rgb);
    fragColor = finalize_output(rgb, alpha);
}