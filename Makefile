ktz.so: ktz.pyx ktzlib/*.h ktzlib/*.c
	python ktz_build.py build_ext --inplace
	rm -rf ktz.c build

clean:
	rm -rf *~ *.pyc *.o

distclean: clean
	rm -f ktz.so termites-ac.net termites-ac.ktz
	rm -rf termites-ac

figs:
	python econet.py termites-ac.rr \
		"full.rebuild" \
		"full.sg.draw" \
		"full.cg.draw" \
		"ring.draw" \
		"ring['R12'].draw"
