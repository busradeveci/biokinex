"""Shared pytest configuration."""

from hypothesis import settings

# Timing deadlines cause random failures on loaded CI machines, and our tests are not about speed.
settings.register_profile("biokinex", deadline=None, max_examples=200)
settings.load_profile("biokinex")
