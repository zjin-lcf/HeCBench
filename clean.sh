for dir in `find . -mindepth 1 -maxdepth 1 -type d | grep -v '.\.git'`
do
	cd ${dir}
	make clean
	cd ..
done
