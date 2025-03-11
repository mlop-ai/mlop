#!/bin/bash

cd $(dirname $0)
WORKDIR="$(realpath .)/.docs"
SRCDIR="$(realpath ..)"

git clone git@github.com:mlop-ai/docs.git $WORKDIR

cd $SRCDIR/docs
magick -background transparent -define 'icon:auto-resize=16,24,32,64' static/img/logo.svg static/img/favicon.ico
npm install
npm run build

cd $WORKDIR
git rm -rf .
git branch -m gh-pages temp
git switch --orphan gh-pages
git branch -D temp
# rsync -av --exclude='node_modules' --exclude='.docusaurus' $SRCDIR/docs/ $WORKDIR
cp -a $SRCDIR/docs/build/* $WORKDIR

git add -A
git commit -m "update"
git push -f origin gh-pages
rm -rf $SRCDIR/docs/build/* $WORKDIR
