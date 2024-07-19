#!/bin/bash/

while getopts n: flag
do
    case "${flag}" in
        n) name=${OPTARG}
    esac
done

echo "Delete feature-environment dataset under: $name? (y/n)"
read ans
if [[ "$ans" -eq y ]]; then 
  echo "delete files.."
  rm -r $name
fi
