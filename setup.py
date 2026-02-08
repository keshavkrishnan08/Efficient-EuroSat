from setuptools import setup, find_packages

setup(
    name='efficient_eurosat',
    version='1.0.0',
    description='Efficient Vision Transformer Attention for Satellite Land Use Classification',
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'numpy>=1.24.0',
        'matplotlib>=3.7.0',
        'seaborn>=0.12.0',
        'pyyaml>=6.0',
        'tqdm>=4.65.0',
        'wandb>=0.15.0',
        'scikit-learn>=1.2.0',
        'pandas>=2.0.0',
        'onnx>=1.14.0',
        'onnxruntime>=1.15.0',
        'Pillow>=9.5.0',
        'pytest>=7.3.0',
    ],
    python_requires='>=3.8',
)
