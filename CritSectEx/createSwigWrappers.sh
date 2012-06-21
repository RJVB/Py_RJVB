#!/bin/sh

case `uname -s` in
	Darwin)
		PLATFORM="__APPLE__"
		;;
	Linux)
		PLATFORM="linux"
		;;
esac

swig -python -D${PLATFORM} CritSectEx.h 
swig -v -python -c++ -D${PLATFORM} msemul.h
