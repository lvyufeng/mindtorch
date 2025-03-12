python setup.py bdist_wheel
rm -rf mindtorch.egg-info
pip uninstall mindtorch -y && pip install dist/*.whl