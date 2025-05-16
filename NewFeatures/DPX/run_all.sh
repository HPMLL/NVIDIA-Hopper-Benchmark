#! /bin/sh

SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"

cd ${SCRIPT_DIR}/bin
for f in ./*; do
    echo ""
    echo "running $f microbenchmark"
    $f
    echo "/////////////////////////////////"
    echo ""
done
