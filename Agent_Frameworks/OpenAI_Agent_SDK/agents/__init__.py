# NOTE: This directory is intentionally NOT used as a Python import package.
#
# The openai-agents SDK installs its own 'agents' module in site-packages.
# Having a local agents/ directory with an __init__.py would shadow that SDK.
#
# All local research agent modules live in pipeline/ instead.
# runner.py and orchestrator.py reorder sys.path so the SDK is found first.
#
# Do not add imports here.
