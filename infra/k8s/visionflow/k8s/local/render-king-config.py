#!/usr/bin/env python3
# run from ./visionflow
import os
from jinja2 import Environment, FileSystemLoader

env = Environment(loader=FileSystemLoader('./k8s/local'))
template = env.get_template('kind-config.yaml.j2')
rendered = template.render(pwd=os.environ["PWD"])
with open('./k8s/local/kind-config.yaml', 'w') as f:
    f.write(rendered)
