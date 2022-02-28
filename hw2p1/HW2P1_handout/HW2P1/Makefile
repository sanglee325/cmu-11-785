autograde:
	mkdir handin
	tar xf handin.tar -C handin
	tar xf autograde.tar
	ls handin
	cp autograde/mytorch_ref/activation_ref.py handin/mytorch/activation.py
	cp autograde/mytorch_ref/linear_ref.py handin/mytorch/linear.py
	cp autograde/mytorch_ref/loss_ref.py handin/mytorch/loss.py
	cp autograde/setup.cfg handin/setup.cfg
	AUTOLAB=1 /usr/local/depot/anaconda3/bin/python3 autograde/runner.py