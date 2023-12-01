#!/bin/bash
# Check if an argument (filename JPG, PNG, BMP, etc) is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <filename (*.jpg,*.bmp,*.png, etc.)>"
    exit 1
fi
REQUIRED_PKG="imagemagick"
PKG_OK=$(dpkg-query -W --showformat='${Status}\n' $REQUIRED_PKG|grep "install ok installed")
echo Checking for $REQUIRED_PKG: $PKG_OK
if [ "" = "$PKG_OK" ]; then
  echo "No $REQUIRED_PKG. Setting up $REQUIRED_PKG."
  sudo apt update && sudo apt-get --yes install $REQUIRED_PKG
fi
convert ${1} -resize 640x480 ${1}.resized.jpg
echo "done"
