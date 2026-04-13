import sys
import types
import numpy as np
import pytest

# We will import from the production module
import cross_platform.qt6_utils.image.gl.texture as tex
from cross_platform.qt6_utils.image.gl.types import GLenum, GLsizei, GLint


class DummyGL:
    # Minimal constants and methods to satisfy calls
    GL_TEXTURE_2D = 0x0DE1
    GL_TEXTURE_BINDING_2D = 0x8069
    GL_UNPACK_ALIGNMENT = 0x0CF5
    GL_CLAMP_TO_EDGE = 0x812F

    GL_TEXTURE_MIN_FILTER = 0x2801
    GL_TEXTURE_MAG_FILTER = 0x2800
    GL_TEXTURE_WRAP_S = 0x2802
    GL_TEXTURE_WRAP_T = 0x2803
    GL_TEXTURE_SWIZZLE_RGBA = 0x8E46

    GL_ACTIVE_TEXTURE = 0x84E0
    GL_TEXTURE0 = 0x84C0

    GL_LINEAR = 0x2601
    GL_NEAREST = 0x2600
    GL_LINEAR_MIPMAP_LINEAR = 0x2703
    GL_NEAREST_MIPMAP_NEAREST = 0x2700

    GL_RED = 0x1903
    GL_GREEN = 0x1904
    GL_BLUE = 0x1905
    GL_ALPHA = 0x1906

    GL_ONE = 1

    GL_R32F = 0x822E
    GL_RGBA8 = 0x8058
    GL_BGRA = 0x80E1
    GL_RGBA = 0x1908
    GL_UNSIGNED_BYTE = 0x1401
    GL_FLOAT = 0x1406

    class GLError(Exception):
        pass

    class GLintMeta(type):
        def __mul__(self, size):
            class Array:
                def __init__(self, *args):
                    self._data = args

                def __iter__(self):
                    return iter(self._data)

                def __getitem__(self, idx):
                    return self._data[idx]

                def __len__(self):
                    return len(self._data)

            return Array

    class GLint(int, metaclass=GLintMeta):
        pass

    class GLuint(int):
        pass

    class GLenum(int):
        pass

    def __init__(self):
        self._bound = 0
        self._active = self.GL_TEXTURE0
        self._params = {}

    # Backend API used by module
    def glPixelStorei(self, pname, param):
        self._params[(pname,)] = int(param)

    def glTexStorage2D(self, target, levels, internalformat, width, height):
        pass

    def glTexImage2D(self, target, level, internalformat, width, height, border, format_, type_, data):
        pass

    def glTexSubImage2D(self, target, level, xoffset, yoffset, width, height, format_, type_, pixels):
        pass

    def glActiveTexture(self, tex):
        self._active = int(tex)

    def glBindTexture(self, target, texture):
        self._bound = int(texture)

    def glGetIntegerv(self, pname):
        """
        Real PyOpenGL function name — glGetIntegerv, not glGetInteger.

        The production module was fixed from glGetInteger (which does not exist
        in PyOpenGL) to glGetIntegerv.  This method implements the correct name.
        """
        pname = int(pname)
        if pname == self.GL_ACTIVE_TEXTURE:
            return self._active
        if pname == self.GL_TEXTURE_BINDING_2D:
            return self._bound
        return 0

    # Keep the old name as a passthrough so any external callers that reference
    # glGetInteger directly do not silently break during the transition.
    glGetInteger = glGetIntegerv

    def glTexParameteriv(self, target, pname, params):
        self._params[(int(target), int(pname))] = tuple(int(x) for x in params)

    def glTexParameteri(self, target, pname, param):
        self._params[(int(target), int(pname))] = int(param)

    def glGenerateMipmap(self, target):
        pass

    def glGenTextures(self, n):
        return 42  # deterministic

    def glDeleteTextures(self, n, arr):
        pass


# Provide GL type wrappers expected by module
class GLHandle(int):
    pass


class GLTexture(int):
    pass


class GLBuffer(int):
    pass


# Patch texture module to use DummyGL and local type wrappers
@pytest.fixture(autouse=True)
def patch_gl(monkeypatch):
    dummy = DummyGL()
    monkeypatch.setattr(tex, "GL", dummy, raising=False)
    monkeypatch.setattr(tex, "GLHandle", GLHandle, raising=False)
    monkeypatch.setattr(tex, "GLTexture", GLTexture, raising=False)
    monkeypatch.setattr(tex, "GLBuffer", GLBuffer, raising=False)

    class DummyCfg:
        USE_IMMUTABLE_STORAGE = True

    monkeypatch.setattr(tex, "get_gl_config", lambda: DummyCfg(), raising=False)

    yield


def test_get_platform_gl_spec(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    spec = tex.get_platform_gl_spec()
    assert int(spec.fmt) == DummyGL.GL_BGRA
    assert spec.swizzle_needed is True

    monkeypatch.setattr(sys, "platform", "linux")
    spec = tex.get_platform_gl_spec()
    assert int(spec.fmt) == DummyGL.GL_RGBA
    assert spec.swizzle_needed is False


def test_ensure_format_compatibility_in_place_bgra():
    spec = tex.TextureSpec(
        internal_format=GLenum(DummyGL.GL_RGBA8),
        fmt=GLenum(DummyGL.GL_BGRA),
        type=GLenum(DummyGL.GL_UNSIGNED_BYTE),
        swizzle_needed=True,
    )
    img = np.zeros((2, 3, 4), dtype=np.uint8)
    img[..., 0] = 10  # R
    img[..., 1] = 20  # G
    img[..., 2] = 30  # B
    img[..., 3] = 40  # A

    out = tex.ensure_format_compatibility(img, spec, in_place=True)
    # In-place swap: R↔B
    assert out is img
    assert np.all(out[..., 0] == 30)   # was B, now in R slot
    assert np.all(out[..., 1] == 20)   # G unchanged
    assert np.all(out[..., 2] == 10)   # was R, now in B slot
    assert np.all(out[..., 3] == 40)   # A unchanged


def test_ensure_format_compatibility_copy_bgra():
    spec = tex.TextureSpec(
        internal_format=GLenum(DummyGL.GL_RGBA8),
        fmt=GLenum(DummyGL.GL_BGRA),
        type=GLenum(DummyGL.GL_UNSIGNED_BYTE),
        swizzle_needed=True,
    )
    img = np.zeros((2, 3, 3), dtype=np.uint8)
    img[..., 0] = 1
    img[..., 1] = 2
    img[..., 2] = 3

    out = tex.ensure_format_compatibility(img, spec, in_place=False)
    assert out is not img
    assert np.all(out[..., 0] == 3)
    assert np.all(out[..., 1] == 2)
    assert np.all(out[..., 2] == 1)


def test_alloc_texture_storage_guard_none_data():
    # Immutable path must not call glTexSubImage2D when data is None.
    tex.alloc_texture_storage(
        target=GLenum(DummyGL.GL_TEXTURE_2D),
        width=GLsizei(2),
        height=GLsizei(2),
        gl_fmt=GLenum(DummyGL.GL_RGBA),
        gl_int_fmt=GLenum(DummyGL.GL_RGBA8),
        gl_type=GLenum(DummyGL.GL_UNSIGNED_BYTE),
        data=None,
        levels=GLint(1),
    )


def test_manager_allocate_upload_and_swizzle():
    mgr = tex.TextureManager()
    key = "k"
    tex_id = mgr.create_texture(key)
    assert int(tex_id) == 42

    spec = tex.TextureSpec(
        internal_format=GLenum(DummyGL.GL_RGBA8),
        fmt=GLenum(DummyGL.GL_RGBA),
        type=GLenum(DummyGL.GL_UNSIGNED_BYTE),
        swizzle_needed=False,
    )

    mgr.allocate_from_spec(key, data=None, width=GLsizei(4), height=GLsizei(4),
                                                         spec=spec)
    state = mgr.get_state(key)
    assert state is not None and state.is_allocated

    img = np.zeros((4, 4, 4), dtype=np.uint8)
    prepared = mgr.upload_image(key, img, spec, in_place=True)
    assert prepared.shape == img.shape

    # RGBA format → swizzle must be RGB (identity mapping)
    assert state.current_swizzle == tex.SwizzleMode.RGB


def test_sampling_mode_sets_params():
    mgr = tex.TextureManager()
    key = "t"
    mgr.create_texture(key)
    with mgr.bound(key):
        tex.TextureManager.set_sampling_mode(
            min_filter=GLenum(DummyGL.GL_NEAREST),
            mag_filter=GLenum(DummyGL.GL_LINEAR),
            wrap_s=GLenum(DummyGL.GL_CLAMP_TO_EDGE),
            wrap_t=GLenum(DummyGL.GL_CLAMP_TO_EDGE),
            generate_mipmaps=False,
        )
    # No exception means all four glTexParameteri calls succeeded.