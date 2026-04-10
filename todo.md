# checklists / todo


## checklist: upload to pypi

1. prepare sources:
    change version number
    commit / merge branch

2. tests:
    in folder "tests", run: "pytest ."

3. generate package:
    (requires pip install build)
    python -m build .

4. upload to pypi:
    (requires pip install twine)
    cd dist
    test file: 
      python -m twine check DISTFILE
    upload: 
      python -m twine upload DISTFILE


      
      
## todo

- remove deprecated "r_core, r_shell ..." interface
- optimize near-field calculation (avoid redundant coefficient evaluation at internal / external positions)

