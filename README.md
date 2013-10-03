CortiCL
=
CortiCL is a C++/OpenCL implementation of the HTM Cortical Learning Algorithm by Numenta. [Whitepaper](http://numenta.org/resources/HTM_CorticalLearningAlgorithms.pdf)

Building
-
The main platform is 64-bit Linux, but it might run elsewhere too. To build:

	mkdir build
	cd build
	cmake ..
	make
	./cldemo

The demo at the moment is a minimal visualization of the Region activation given a somewhat random spatio-temporal pattern seen on the left. More demos are planned.

Use of the algorithm
-

Numenta owns the Intellectual Property of the HTM algorithm. Please refer to Numenta website (http://numenta.org/) for more information.

Also note that I am not affiliated with the company, and that this library is the result of programming in my free time.
