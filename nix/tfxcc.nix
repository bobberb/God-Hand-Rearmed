# tfxcc (tools/linux/tfxcc): the Vulkan renderer's shader compile as one program, on the shaderc, glslang and
# SPIRV-Tools the game ships (nix/deps.nix, thirdparty/pcsx2-gs/shaderc.cmake). For the tfx-variants check.
# Its own narrow source set, so only a change to the compiler's inputs rebuilds shaderc.
{ pkgs, lib, deps }:
let
  src = lib.fileset.toSource {
    root = ../.;
    fileset = lib.fileset.unions [ ../tools/linux/tfxcc ../thirdparty/pcsx2-gs/shaderc.cmake ];
  };
  fetchContentFlags = lib.mapAttrsToList (name: path: "-DFETCHCONTENT_SOURCE_DIR_${name}=${path}") deps.fetchContent;
in
pkgs.stdenv.mkDerivation {
  pname = "tfxcc";
  version = "shaderc-2026.2";
  inherit src;
  nativeBuildInputs = with pkgs; [ cmake ninja python3 ];
  dontUseCmakeConfigure = true;
  configurePhase = ''
    cmake -S tools/linux/tfxcc -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DFETCHCONTENT_FULLY_DISCONNECTED=ON ${lib.concatStringsSep " " fetchContentFlags}
  '';
  buildPhase = "ninja -C build -j\${NIX_BUILD_CORES:-4} tfxcc";
  installPhase = "install -Dm755 build/tfxcc $out/bin/tfxcc";
  meta.description = "Compiles a GLSL source as GHR's Vulkan renderer does (libshaderc, Vulkan 1.0, -O)";
}
