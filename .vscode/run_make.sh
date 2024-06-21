#!/bin/bash
set -e

TARGET=$1
if [ -z "$TARGET" ]; then
    echo "Usage: $0 <target>"
    exit 1
fi

if [ ! -f "Makefile" ]; then
    echo "No Makefile found in $PWD"
    exit 1
fi

for t in $TARGET; do
    if [ $t == "all" ]; then
        CMD="bear -- make all"
    else
        CMD="make $t"
    fi
    echo "Building target \"$t\" in \"$PWD\" ..."
    echo "# $CMD"
    eval $CMD
    echo "Target \"$t\" done."
    echo
done

COMPILE_COMMANDS="compile_commands.json"
if [ ! -f "$COMPILE_COMMANDS" ]; then
    echo "No $COMPILE_COMMANDS generated in $PWD"
fi
