// Offline descriptor-tracking check for one dumped shader (.cap or .reg). Exits 0 on success; a
// translation failure exits through EXIT with its message.
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/shader.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

using namespace Libs::Graphics;

int main(int argc, char** argv) {
	if (argc != 2) return 2;
	auto* file = std::fopen(argv[1], "rb");
	if (file == nullptr) return 2;
	std::vector<uint32_t> data;
	uint32_t word = 0;
	while (std::fread(&word, 4, 1, file) == 1) data.push_back(word);
	std::fclose(file);
	if (data.size() < 4) return 2;
	ShaderRecompiler::CompileOptions options;
	options.dump_ir = std::getenv("SHADER_CHECK_DUMP") != nullptr;
	auto vertex  = std::make_unique<ShaderVertexInputInfo>();
	auto pixel   = std::make_unique<ShaderPixelInputInfo>();
	auto compute = std::make_unique<ShaderComputeInputInfo>();
	std::vector<uint32_t> user, code, back;
	size_t at = 0;
	const auto take = [&](std::vector<uint32_t>& out, size_t count) {
		if (at + count > data.size()) std::_Exit(2);
		out.assign(data.begin() + at, data.begin() + at + count);
		at += count;
	};
	const auto input = [&](void* out, size_t size) {
		if (size != data[5] || at * 4 + size > data.size() * 4) std::_Exit(2);
		std::memcpy(out, data.data() + at, size);
		at += (size + 3) / 4;
	};
	if (data[0] == 0x5041434bu && data.size() > 10) {
		options.stage          = static_cast<ShaderType>(data[1]);
		options.wave_size      = data[2];
		options.user_data_base = data[3];
		options.shader_hash    = data[8] | uint64_t {data[9]} << 32u;
		at                     = 10;
		if (options.stage == ShaderType::Compute) input(compute.get(), sizeof(*compute));
		else if (options.stage == ShaderType::Pixel) input(pixel.get(), sizeof(*pixel));
		else input(vertex.get(), sizeof(*vertex));
		take(user, data[4]);
		take(code, data[6]);
		take(back, data[7]);
	} else if (data[0] == 0x4745524bu) {
		std::vector<uint32_t> regs;
		const auto type = data[1], count = data[3], bytes = data[2];
		at = 4;
		take(regs, size_t {count} * 2u);
		take(code, bytes / 4u);
		// A trailing word is the header's exact scratch size. Dumps without one only have SCRATCH_EN,
		// so they get the header field's maximum; only whether it is zero matters to translation.
		const bool     has_scratch_size = at < data.size();
		const uint32_t dumped_scratch   = has_scratch_size ? data[at] : 0u;
		const auto     scratch_size     = [&](uint32_t rsrc2) {
			return has_scratch_size ? dumped_scratch : (rsrc2 & 1u) != 0u ? 0xffffu : 0u;
		};
		const auto reg = [&](uint32_t offset) {
			for (uint32_t i = 0; i < count; ++i)
				if (regs[i * 2u] == offset) return regs[i * 2u + 1u];
			return 0u;
		};
		if (type == 0u) { // compute
			const auto rsrc2 = reg(0x213);
			options.stage    = ShaderType::Compute;
			const auto users = (rsrc2 >> 1u) & 31u;
			user.resize(users);
			compute->scratch_size_dwords = scratch_size(rsrc2);
			compute->group_id[0] = (rsrc2 >> 7u) & 1u;
			compute->group_id[1] = (rsrc2 >> 8u) & 1u;
			compute->group_id[2] = (rsrc2 >> 9u) & 1u;
			compute->tg_size_en  = (rsrc2 >> 10u) & 1u;
			compute->thread_ids_num     = static_cast<int>((rsrc2 >> 11u) & 3u) + 1;
			compute->workgroup_register = static_cast<int>(users);
			for (uint32_t i = 0; i < 3u; ++i)
				compute->threads_num[i] = std::max(reg(0x207 + i) & 0xffffu, 1u);
			compute->lds_size_dwords = ((rsrc2 >> 15u) & 0x1ffu) * 128u;
			// The wave size is a dispatch modifier, so the caller tries both.
			const auto* wave = std::getenv("SHADER_CHECK_WAVE");
			compute->wave_size = wave != nullptr && wave[0] == '3' ? 32u : 64u;
			options.wave_size  = compute->wave_size;
		} else if (type == 1u) { // pixel
			const auto rsrc2 = reg(0x0b);
			options.stage    = ShaderType::Pixel;
			user.resize(((rsrc2 >> 1u) & 31u) | ((rsrc2 >> 27u) & 1u) << 5u);
			pixel->scratch_size_dwords  = scratch_size(rsrc2);
			pixel->input_num            = 32u;
			pixel->ps_system_input_base = 2u;
			const auto* wave = std::getenv("SHADER_CHECK_WAVE");
			options.wave_size = pixel->wave_size = wave != nullptr && wave[0] == '3' ? 32u : 64u;
		} else {
			std::puts("SKIP stage");
			return 3;
		}
	} else {
		return 2;
	}
	if (options.stage == ShaderType::Compute) options.input_info.compute = compute.get();
	else if (options.stage == ShaderType::Pixel) options.input_info.pixel = pixel.get();
	else options.input_info.vertex = vertex.get();
	options.user_data = user;
	options.back_code = back;
	const auto source = ShaderRecompiler::PrepareShaderSource(code, options);
	if (source.unsupported_call_pc != UINT32_MAX || !source.calls.empty()) {
		std::puts("SKIP calls");
		return 3;
	}
	const auto result = ShaderRecompiler::TranslateProgram(source.decoded, options);
	std::puts("OK");
	std::fflush(nullptr);
	std::_Exit(0);
}
