#include "common/assert.h"

#include <gtest/gtest.h>

#include <csetjmp>
#include <cstring>

namespace {

TEST(KeepGoing, SnapshotDiagnosticBeforeLongjmp) {
	std::jmp_buf recover;
	auto*        previous = Common::g_dbg_recover;
	if (setjmp(recover) == 0) {
		Common::g_dbg_recover = &recover;
		Common::DbgExitHandler("test.cpp", 42, "shader blew up");
		FAIL() << "DbgExitHandler should longjmp when g_dbg_recover is set";
	}
	Common::g_dbg_recover = previous;

	const auto& diag = Common::LastFatalDiagnostic();
	EXPECT_STREQ(diag.text, "shader blew up");
	EXPECT_STREQ(diag.file, "test.cpp");
	EXPECT_EQ(diag.line, 42);
}

TEST(KeepGoing, RecoveryPointerClearedBeforeLongjmp) {
	std::jmp_buf recover;
	auto*        previous = Common::g_dbg_recover;
	if (setjmp(recover) == 0) {
		Common::g_dbg_recover = &recover;
		Common::DbgExitHandler("a.cpp", 1, "boom");
		FAIL() << "should not return";
	}
	EXPECT_EQ(Common::g_dbg_recover, nullptr);
	Common::g_dbg_recover = previous;
}

TEST(KeepGoing, NestedRecoveryRestoresPreviousPointer) {
	std::jmp_buf outer;
	std::jmp_buf inner;
	auto*        previous = Common::g_dbg_recover;

	ASSERT_EQ(setjmp(outer), 0);
	Common::g_dbg_recover = &outer;

	if (setjmp(inner) == 0) {
		Common::g_dbg_recover = &inner;
		Common::DbgExitHandler("inner.cpp", 7, "inner failure");
		FAIL() << "should not return";
	}
	// Inner recovery took the longjmp; caller restores outer.
	Common::g_dbg_recover = &outer;
	const auto& diag = Common::LastFatalDiagnostic();
	EXPECT_STREQ(diag.text, "inner failure");
	EXPECT_STREQ(diag.file, "inner.cpp");

	Common::g_dbg_recover = previous;
}

} // namespace

int main(int argc, char** argv) {
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
