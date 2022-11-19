### WIP

FROM continuumio/miniconda3

# Add environment.yml to the build context and create the environment
ARG conda_env=base
# ADD environment.yml /tmp/environment.yml
# RUN conda env create -f /tmp/environment.yml

# Activating the environment and starting the jupyter notebook
RUN echo "source activate ${conda_env}" > ~/.bashrc
ENV PATH /opt/conda/envs/${conda_env}/bin:$PATH

RUN conda install mamba -n base -c conda-forge

RUN mamba install numpy pandas scikit-learn jupyterlab pytorch torchvision pytorch-cuda=11.7 -c pytorch -c nvidia
RUN pip3 install kaggle albumentations 

# # Start jupyter server on container
# EXPOSE 8888
# ENTRYPOINT ["jupyter","lab","--ip=0.0.0.0", \
#     "--port=8888","--allow-root","--no-browser"]