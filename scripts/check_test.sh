#!/bin/bash
set -ex

if [[ -z ${BRANCH} ]];then
	BRANCH="master"
fi

BENCHMARK_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")/.." && pwd )"
echo ${BENCHMARK_ROOT}

function check_style(){
	trap 'abort' 0
	pip install cpplint pylint pytest astroid isort
	pre-commit install
	commit_files=on
    	for file_name in `git diff --numstat | awk '{print $NF}'`;do
        	if [ ! pre-commit run --files $file_name ]; then
            		git diff
            		commit_files=off
        	fi
    	done

    	if [ $commit_files == 'off' ];then
        	echo "code format error"
        	exit 1
    	fi
    	trap 0
}

function main(){
	local CMD=$1
	case $CMD in
		check_style)
			check_style
		;;
		*)
		echo "Failed"
		exit 1
		;;
	esac
}
main $@
