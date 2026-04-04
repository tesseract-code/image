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
//   5. Gamma correction
//   6. Contrast & brightness (tone mapping)
//   7. Color balance (after gamma)
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
// lut_type:  0 = Linear (pass-through)
//            1 = Logarithmic
//            2 = Square-root
//            3 = Square
uniform bool  lut_enabled;
uniform float lut_norm_factor;  // 1.0 / (lut_max - lut_min)
uniform float lut_min;
uniform int   lut_type;


// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const vec3 BT709_LUMINANCE = vec3(0.2126, 0.7152, 0.0722);
const float EPSILON = 1e-5;
const float GAMMA_THRESHOLD = 0.001;


// ---------------------------------------------------------------------------
// Transfer Functions — all map [0, 1] → [0, 1]
// ---------------------------------------------------------------------------

// Log: compress bright end.
// log(x + 1.0) avoids singularity at 0.
// log(2.0) is the value at x=1, so dividing by it maps [0,1] -> [0,1] exactly.
float lut_log(float x) {
    return log(x + 1.0) / log(2.0);
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

// Sample texture
vec4 sample_texture() {
    return texture(imageTexture, fragTexCoord);
}

// Normalize RGB to [0, 1] range
vec3 normalize_range(vec3 rgb) {
    float range = norm_vmax - norm_vmin;
    if (abs(range) > EPSILON) {
        rgb = (rgb - vec3(norm_vmin)) / range;
    }
    return rgb;
}

// Extract luminance for intensity-based processing
float extract_luminance(vec3 rgb) {
    // For monochrome inputs R == G == B; dot product degenerates to rgb.r.
    // For RGB inputs we want a luminance scalar before the data transform.
    // Use BT.709 coefficients for perceptually uniform intensity.
    return dot(rgb, BT709_LUMINANCE);
}

// Apply LUT transform to intensity
float apply_intensity_transform(float intensity) {
    intensity = clamp(intensity, 0.0, 1.0);
    if (lut_enabled) {
        intensity = apply_lut(intensity);
    }
    return intensity;
}

// Apply colormap or reconstruct RGB from transformed intensity
vec3 apply_colormap(vec3 rgb, float intensity) {
    if (use_cmap) {
        // Replace rgb entirely via colormap lookup
        vec4 lutColor = texture(colormapTexture, vec2(intensity, 0.5));
        return lutColor.rgb;
    } else if (lut_enabled) {
        // No colormap, but a data transform was requested — reconstruct
        // rgb from the transformed intensity, preserving any hue ratio.
        // Avoid divide-by-zero on black pixels.
        float lum = dot(rgb, BT709_LUMINANCE);
        return (lum > EPSILON) ? rgb * (intensity / lum) : vec3(intensity);
    }
    return rgb;
}

// Apply contrast and brightness adjustments
vec3 apply_contrast_brightness(vec3 rgb) {
    // Clamp inputs first — upstream ops can push values outside [0, 1]
    // and unbounded values interact badly with extreme contrast settings.
    rgb = clamp(rgb, 0.0, 1.0);
    rgb = (rgb - 0.5) * contrast + 0.5 + brightness;
    return rgb;
}

// Apply gamma correction
vec3 apply_gamma(vec3 rgb) {
    // Guard against negative base — pow() on negative values is undefined
    // in GLSL for non-integer exponents.
    if (abs(inv_gamma - 1.0) > GAMMA_THRESHOLD) {
        rgb = pow(max(rgb, vec3(0.0)), vec3(inv_gamma));
    }
    return rgb;
}

// Apply color balance
vec3 apply_color_balance(vec3 rgb) {
    return rgb * color_balance;
}

// Apply inversion
vec3 apply_invert(vec3 rgb) {
    if (invert) {
        rgb = 1.0 - rgb;
    }
    return rgb;
}

// Final output clamping
vec4 finalize_output(vec3 rgb, float alpha) {
    return vec4(clamp(rgb, 0.0, 1.0), alpha);
}

// Check if we can skip processing (early exit optimization)
bool should_skip_processing(vec3 rgb, float alpha) {
    return !use_cmap
        && !lut_enabled
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

    if (should_skip_processing(rgb, alpha)) {
        fragColor = texColor;
        return;
    }

    rgb = normalize_range(rgb);

    // LUT / Colormap (in linear space)
    if (use_cmap || lut_enabled) {
        float intensity = extract_luminance(rgb);
        intensity = apply_intensity_transform(intensity);
        rgb = apply_colormap(rgb, intensity);
    }

    rgb = apply_gamma(rgb);
    rgb = apply_contrast_brightness(rgb);
    rgb = apply_color_balance(rgb);
    rgb = apply_invert(rgb);
    fragColor = finalize_output(rgb, alpha);
}