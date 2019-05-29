1. data folder
	- caption and question에 caption, question 폴더 생성
	- caption : 2014 train, val data 넣기
	- question : GQA question dataset 넣기

	- old_data 폴더 : visual dialog로 생성했던 질문 및 keyword 데이터셋 넣기.

2. result folder
	- caption, answer, id, question, train 폴더 생성하기.

3. 실행
	-  integrate_graph : 모든 관계 그래프들을 하나의 그래프로 통합함.
	- intgerate_qeustion : 기존의 keyword, question 데이터셋에 GQA 데이터셋 질문들을 통합함.