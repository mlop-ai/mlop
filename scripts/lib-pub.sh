#!/bin/bash

cd $(dirname $0)
WORKDIR="$(realpath .)/.pypi"
SRCDIR="$(realpath ..)"
mkdir -p $WORKDIR

: << "GIT"
git clone git@github.com:mlop-ai/mlop.git $WORKDIR
cd $WORKDIR; git rm -rf .; cd ..
GIT

cd $SRCDIR
cp -a mlop pyproject.toml .gitignore README.md LICENSE $WORKDIR
cd $WORKDIR
VERSION=$(grep '__version__ = ' ./mlop/__init__.py | sed -E 's/__version__ = "(.*)"/\1/')

: << "GIT"
git add -A  # fixme: support deletion
read -r -p "Confirm push v$VERSION? [y/N]" RES
case "$RES" in
    [yY][eE][sS]|[yY]) 
        git commit -m "v$VERSION"
        git push
        ;;
    *)
        rm -rf $WORKDIR
        exit 1
        ;;
esac
GIT

read -r -p "Confirm publish v$VERSION? [y/N]" RES
case "$RES" in
    [yY][eE][sS]|[yY]) 
        # python -m pip install build twine
        python -m build
        twine upload dist/*
        ;;
    *)
        rm -rf $WORKDIR
        exit 1
        ;;
esac

rm -rf $WORKDIR
