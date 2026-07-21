@echo off
cd /d "t:\download\Drug--Induced-Cardiotoxicity--main\Drug--Induced-Cardiotoxicity--main"
call "t:\download\Drug--Induced-Cardiotoxicity--main\.venv\Scripts\activate.bat"
python counterfactual.py --num_molecules 10 --num_variants 2
pause
