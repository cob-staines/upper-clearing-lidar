set lista=A B C D

for %%a in (%lista%) do (
	echo %%a > C:\Users\Cob\Desktop\thefile.txt
	echo/ > C:\Users\Cob\Desktop\thefile.txt
	) 