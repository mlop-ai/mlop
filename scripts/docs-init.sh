#!/bin/bash

cd $(dirname $0)
SITE_DIR="docusaurus"
DOCS_DIR="$(realpath ..)/docs/"
cd $DOCS_DIR
npm init docusaurus@latest $SITE_DIR classic

cd $SITE_DIR
npm run start