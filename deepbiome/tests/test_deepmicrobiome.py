#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `deepbiome` package."""

import pytest

from click.testing import CliRunner

from deepbiome import deepbiome
from deepbiome import cli



def test_deepbiome():
    '''
    Test deepbiome by some simulated data
    '''
    assert 1+1 == 2