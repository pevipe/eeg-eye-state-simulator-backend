# Copyright 2024 Pelayo Vieites Pérez
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#

import os

base_directory = os.environ.get('SIMULATOR_PERSISTENT_PATH')

for folder in ['datasets', 'hyperparams', 'results']:
    path = os.path.join(base_directory, folder)
    if not os.path.exists(path):
        os.makedirs(path)
