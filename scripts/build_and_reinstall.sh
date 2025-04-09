pip uninstall torch torch_npu -y

rm -rf ./build
rm -rf ./dist
python setup.py bdist_wheel
rm -rf *.egg-info
pip uninstall mindtorch -y && pip install dist/*.whl