# Read OpenCL code from source file and re-write it as a C string to the given file

file(READ ${SOURCE} CONTENTS)

string(REPLACE "\n" "\\n\\\n" CONTENTS "${CONTENTS}")
set(CONTENTS "\"" ${CONTENTS} "\";\n")

file(WRITE ${DESTINATION} "${CONTENTS}")
