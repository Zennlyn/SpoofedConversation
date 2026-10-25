#!/bin/bash
set -x

FILES=("label_R01TTS.0.a.txt" "label_R01TTS.0.b.txt" "metadata_crossfade.csv" "R01TTS.0.a.tgz" "R01TTS.0.b.tgz" "README.txt")
BASE_URL="https://zenodo.org/records/14214149/files"
DEST_DIR="."

mkdir -p "${DEST_DIR}"

for file in "{FILES[@]}"; do
	link="${BASE_URL}/${file}?download=1"
	dest_path="${DEST_DIR}/${file}"
	
	if [ ! -f "${dest_path}" ]; then
		echo "Download ${file}..."
		if [ "${file}" == "R01TTS.0.b.tgz" ] || [ "${file}" == "R01TTS.0.a.tgz" ]; then
			wget -q --show-progress -c "${link}" - O "${DEST_DIR}/${file}"
			tar -zxvf "${DEST_DIR}/${file}" -C "${DEST_DIR}"
			rm "${DEST_DIR}/${file}"
		else
			wget -q --show-progress -c "${link}" -O "${dest_path}"
		fi
	else
		echo "${file} already exists, skipping."
	fi
done

echo "LlamaPartialSpoof is downloaded"
