/* tfxcc: compile one GLSL source from stdin the way the Vulkan renderer does (VKShaderCache::CompileShaderToSPV,
 * thirdparty/pcsx2/pcsx2/GS/Renderers/Vulkan/VKShaderCache.cpp, its non-debug path): the same libshaderc, the same
 * options, input name and entry point. For tools/linux/check_tfx_variants.py.
 *
 *   tfxcc vert|frag [debug] < source.glsl   exit 0 compiled, 1 failed (shaderc's messages on stderr), 2 bad usage
 *
 * debug: the GSConfig.UseDebugDevice path, no optimisation and debug info (stock shaderc, as
 * thirdparty/pcsx2-gs/shaderc.cmake builds it: no SHADERC_PCSX2_CUSTOM).
 */
#include <shaderc/shaderc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char** argv)
{
	shaderc_shader_kind kind;
	const int debug = argc == 3 && strcmp(argv[2], "debug") == 0;
	if ((argc == 2 || debug) && strcmp(argv[1], "vert") == 0)
		kind = shaderc_glsl_vertex_shader;
	else if ((argc == 2 || debug) && strcmp(argv[1], "frag") == 0)
		kind = shaderc_glsl_fragment_shader;
	else
	{
		fprintf(stderr, "usage: tfxcc vert|frag [debug] < source.glsl\n");
		return 2;
	}

	size_t cap = 1 << 16, len = 0;
	char* src = malloc(cap);
	for (size_t n; src && (n = fread(src + len, 1, cap - len, stdin)) > 0;)
	{
		if ((len += n) == cap)
		{
			char* bigger = realloc(src, cap *= 2);
			if (!bigger)
				free(src);
			src = bigger;
		}
	}
	if (!src)
		return 2;

	shaderc_compiler_t compiler = shaderc_compiler_initialize();
	shaderc_compile_options_t options = shaderc_compile_options_initialize();
	shaderc_compile_options_set_source_language(options, shaderc_source_language_glsl);
	shaderc_compile_options_set_target_env(options, shaderc_target_env_vulkan, 0);
	if (debug)
		shaderc_compile_options_set_generate_debug_info(options);
	shaderc_compile_options_set_optimization_level(
		options, debug ? shaderc_optimization_level_zero : shaderc_optimization_level_performance);

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
