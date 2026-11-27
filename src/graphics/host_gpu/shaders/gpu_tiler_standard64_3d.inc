#include "gpu_tiler_standard4_3d.inc"
uvec3 standard64_3d_extent() {
	if (ELEMENT_BYTES == 1) return uvec3(64, 32, 32);
	if (ELEMENT_BYTES == 2) return uvec3(32, 32, 32);
	if (ELEMENT_BYTES == 4) return uvec3(32, 32, 16);
	if (ELEMENT_BYTES == 8) return uvec3(32, 16, 16);
	return uvec3(16, 16, 16);
}
uint standard64_3d_offset(uvec3 p) {
	// RADV miscompiles the variable shift amounts from the `s` vector for this specialization.
	// Constant bit selects are equivalent and compile correctly.
	uint offset = standard3d_offset(p);
	if (ELEMENT_BYTES == 1) {
		return offset ^ ((p.x & 16u) << 8) ^ ((p.z & 16u) << 9) ^
		       ((p.y & 16u) << 10) ^ ((p.x & 32u) << 10);
	}
	if (ELEMENT_BYTES == 2) {
		return offset ^ ((p.x & 8u) << 9) ^ ((p.z & 16u) << 9) ^
		       ((p.y & 16u) << 10) ^ ((p.x & 16u) << 11);
	}
	if (ELEMENT_BYTES == 4) {
		return offset ^ ((p.x & 8u) << 9) ^ ((p.z & 8u) << 10) ^
		       ((p.y & 16u) << 10) ^ ((p.x & 16u) << 11);
	}
	if (ELEMENT_BYTES == 8) {
		return offset ^ ((p.x & 8u) << 9) ^ ((p.z & 8u) << 10) ^
		       ((p.y & 8u) << 11) ^ ((p.x & 16u) << 11);
	}
	return offset ^ ((p.x & 4u) << 10) ^ ((p.z & 8u) << 10) ^
	       ((p.y & 8u) << 11) ^ ((p.x & 8u) << 12);
}
