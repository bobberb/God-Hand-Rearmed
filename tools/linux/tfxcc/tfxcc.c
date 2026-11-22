/* tfxcc: compile one GLSL source from stdin the way the Vulkan renderer does (VKShaderCache::CompileShaderToSPV,
 * thirdparty/pcsx2/pcsx2/GS/Renderers/Vulkan/VKShaderCache.cpp, its non-debug path): the same libshaderc, the same
 * options, input name and entry point. For tools/linux/check_tfx_variants.py.
 *
 *   tfxcc vert|frag < source.glsl     exit 0 compiled, 1 failed (shaderc's messages on stderr), 2 bad usage
 */
#include <shaderc/shaderc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char** argv)
{
	shaderc_shader_kind kind;
	if (argc == 2 && strcmp(argv[1], "vert") == 0)
		kind = shaderc_glsl_vertex_shader;
	else if (argc == 2 && strcmp(argv[1], "frag") == 0)
		kind = shaderc_glsl_fragment_shader;
	else
	{
		fprintf(stderr, "usage: tfxcc vert|frag < source.glsl\n");
		return 2;
	}

	size_t cap = 1 << 16, len = 0;
	char* src = malloc(cap);
	for (size_t n; src && (n = fread(src + len, 1, cap - len, stdin)) > 0;)
		if ((len += n) == cap)
			src = realloc(src, cap *= 2);
	if (!src)
		return 2;

	shaderc_compiler_t compiler = shaderc_compiler_initialize();
	shaderc_compile_options_t options = shaderc_compile_options_initialize();
	shaderc_compile_options_set_source_language(options, shaderc_source_language_glsl);
	shaderc_compile_options_set_target_env(options, shaderc_target_env_vulkan, 0);
	shaderc_compile_options_set_optimization_level(options, shaderc_optimization_level_performance);

	shaderc_compilation_result_t result = shaderc_compile_into_spv(compiler, src, len, kind, "source", "main", options);
	const shaderc_compilation_status status =
		result ? shaderc_result_get_compilation_status(result) : shaderc_compilation_status_null_result_object;
	if (status != shaderc_compilation_status_success)
		fprintf(stderr, "status %d\n%s", (int)status, result ? shaderc_result_get_error_message(result) : "null result\n");

	shaderc_result_release(result);
	shaderc_compile_options_release(options);
	shaderc_compiler_release(compiler);
	free(src);
	return status == shaderc_compilation_status_success ? 0 : 1;
}
