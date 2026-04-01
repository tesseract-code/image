#version 410 core

// ---------------------------------------------------------------------------
// image.frag
//
// Processing pipeline:
//
//   1. Texture sample
//   2. Normalization        [vmin, vmax] → [0, 1]
//   3. LUT / data transform  log | sqrt | square  (on scalar intensity)
//   4. Colormap              intensity → RGB via LUT texture
//   5. Contrast & brightness
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
// lut_type:  0 = Linear (pass-through)
//            1 = Logarithmic
//            2 = Square-root
//            3 = Square
uniform bool  lut_enabled;
uniform float lut_norm_factor;  // 1.0 / (lut_max - lut_min)
uniform float lut_min;
uniform int   lut_type;


// ---------------------------------------------------------------------------
// Transfer functions — all map [0, 1] → [0, 1]
// ---------------------------------------------------------------------------

// Log: compress bright end.
// log(x + 1.0) avoids singularity at 0.
// log(2.0) is the value at x=1, so dividing by it maps [0,1] -> [0,1] exactly.
// Avoid log1p — not reliably available on macOS GLSL drivers.
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

// Dispatch helper — keeps main() readable.
float apply_lut(float x) {
    // Remap to [0, 1] using the dataset's own min/max before the transform.
    x = clamp((x - lut_min) * lut_norm_factor, 0.0, 1.0);

    if      (lut_type == 1) x = lut_log(x);
    else if (lut_type == 2) x = lut_sqrt(x);
    else if (lut_type == 3) x = lut_square(x);
    // lut_type == 0: linear, x unchanged.

    return x;  // already in [0, 1]
}


// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
void main() {

    // ------------------------------------------------------------------
    // 1. Sample texture
    // ------------------------------------------------------------------
    vec4 texColor = texture(imageTexture, fragTexCoord);
    vec3 rgb      = texColor.rgb;
    float alpha   = texColor.a;

    // ------------------------------------------------------------------
    // 2. Normalisation  →  [0, 1]
    // ------------------------------------------------------------------
    float range = norm_vmax - norm_vmin;
    if (abs(range) > 1e-5) {
        rgb = (rgb - vec3(norm_vmin)) / range;
    }

    // ------------------------------------------------------------------
    // 3. LUT / data transform  (scalar, before colormap)
    //
    //    We work on a single intensity value here so the transfer function
    //    acts on data magnitude, not on independent RGB channels.  Applying
    //    it after colormapping would transform already-perceptual hues, which
    //    is meaningless for log/sqrt/square stretches.
    // ------------------------------------------------------------------
    float intensity;

    if (use_cmap || lut_enabled) {
        // For monochrome inputs R == G == B; dot product degenerates to rgb.r.
        // For RGB inputs we want a luminance scalar before the data transform.
        // Use BT.709 coefficients.
        intensity = dot(rgb, vec3(0.2126, 0.7152, 0.0722));
        intensity = clamp(intensity, 0.0, 1.0);

        if (lut_enabled) {
            intensity = apply_lut(intensity);
        }
    }

    // ------------------------------------------------------------------
    // 4. Colormap lookup
    //    Replaces rgb entirely; alpha is preserved from original texture.
    // ------------------------------------------------------------------
    if (use_cmap) {
        vec4 lutColor = texture(colormapTexture, vec2(intensity, 0.5));
        rgb = lutColor.rgb;
    } else if (lut_enabled) {
        // No colormap, but a data transform was requested — reconstruct
        // rgb from the transformed intensity, preserving any hue ratio.
        // Avoid divide-by-zero on black pixels.
        float lum = dot(rgb, vec3(0.2126, 0.7152, 0.0722));
        rgb = (lum > 1e-5) ? rgb * (intensity / lum) : vec3(intensity);
    }

    // ------------------------------------------------------------------
    // 5. Contrast & brightness
    //    Clamp inputs first — upstream ops can push values outside [0, 1]
    //    and unbounded values interact badly with extreme contrast settings.
    // ------------------------------------------------------------------
    rgb = clamp(rgb, 0.0, 1.0);
    rgb = (rgb - 0.5) * contrast + 0.5 + brightness;

    // ------------------------------------------------------------------
    // 6. Gamma correction
    //    Guard against negative base — pow() on negative values is undefined
    //    in GLSL for non-integer exponents.
    // ------------------------------------------------------------------
    if (abs(inv_gamma - 1.0) > 0.001) {
        rgb = pow(max(rgb, vec3(0.0)), vec3(inv_gamma));
    }

    // ------------------------------------------------------------------
    // 7. Color balance
    // ------------------------------------------------------------------
    rgb *= color_balance;

    // ------------------------------------------------------------------
    // 8. Invert
    // ------------------------------------------------------------------
    if (invert) {
        rgb = 1.0 - rgb;
    }

    // ------------------------------------------------------------------
    // 9. Final output — clamp to display range, restore original alpha
    // ------------------------------------------------------------------
    fragColor = vec4(clamp(rgb, 0.0, 1.0), alpha);
}
