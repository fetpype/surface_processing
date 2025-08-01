
FROM pymesh/pymesh:py3.7


WORKDIR /opt/packages/

ADD https://api.github.com/repos/fetpype/surface_processing/git/refs/heads/main version.json

RUN git clone https://github.com/fetpype/surface_processing.git

WORKDIR /opt/packages/surface_processing

RUN pip install -r requirements.txt

#ENTRYPOINT ["python"]


