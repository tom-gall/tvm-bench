#!/bin/sh

. ../../lib/sh-test-lib
OUTPUT="$(pwd)/output"

usage() {
    echo "Usage: $0 [-s <true|false>]" 1>&2
    exit 1
}

while getopts "s:a:d:" o; do
  case "$o" in
    s) SKIP_INSTALL="${OPTARG}" ;;
    a) ARMNN_TARBALL="${OPTARG}" ;;
    d) TEST_DIR="${OPTARG}" ;;
    *) usage ;;
  esac
done

! check_root && error_msg "You need to be root to run this script."
create_out_dir "${OUTPUT}"

pkgs="ntp wget gcc g++ python3 python3-pip"
dhclient
install_deps "${pkgs}" "${SKIP_INSTALL}"

if [[-z ${ARMNN_TARBALL} ]]; then
    mkdir -p ${TEST_DIR}
    cd ${TEST_DIR}
    wget -O armnn.tar.xz "${ARMNN_TARBALL}"
    tar xf armnn.tar.xz
    export BASEDIR="/home/buildslave/workspace/armnn-ci-build"
fi

if [[ ${SKIP_INSTALL} == false ]]; then
    cd $BASEDIR/build
    ln -s libprotobuf.so.15.0.0 ./libprotobuf.so.15
    export LD_LIBRARY_PATH=`pwd`
    chmod a+x UnitTests
fi
lava-test-case ArmNN-Unit-Tests --shell ./UnitTests
./UnitTests