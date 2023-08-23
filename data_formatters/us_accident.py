# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Link to repository: https://github.com/google-research/google-research/tree/master/tft
# Lint as: python3

from Utils.base import DataTypes, InputTypes
from data_formatters.general import GeneralFormatter

DataFormatter = GeneralFormatter


class USAccidentFormatter(DataFormatter):

    _column_definition = [
        ('seconds_from_start', DataTypes.REAL_VALUED, InputTypes.TIME),
        ('id', DataTypes.REAL_VALUED, InputTypes.ID),
        ('Severity', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
        ('target', DataTypes.CATEGORICAL, InputTypes.TARGET),
        ('categorical_id', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
    ]

    def __init__(self, pred_len):
        super(USAccidentFormatter, self).__init__(pred_len)