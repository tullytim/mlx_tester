#!/bin/bash
set -e
swift build
cp .build/arm64-apple-macosx/debug/MLXTester .build/MLXTester.app/Contents/MacOS/MLXTester
open .build/MLXTester.app
