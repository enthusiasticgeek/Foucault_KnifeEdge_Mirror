#!/bin/bash

if [ -z "$1" ]; then
   echo -n -e "commit message missing"
   exit 1
fi
GIT_COMMITTER_NAME="Pratik Tambe" GIT_COMMITTER_EMAIL="enthusiasticgeek@gmail.com" git commit --author="Pratik Tambe <enthusiasticgeek@gmail.com>" -m "${1}"
