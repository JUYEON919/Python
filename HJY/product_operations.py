# 문제 : 쇼핑몰 제품 관리 시스템
# 문제 설명:
#   쇼핑몰의 제품 관리 시스템을 구현하는 문제입니다. 
#   첫 번째 파일은 Product 클래스를 정의하고, 
#   두 번째 파일은 Shop 클래스를 작성하여 여러 제품을 관리합니다.

class Product:
    def __init__(self, name, price, quantity):
        self.name = name
        self.price = price
        self.quantity = quantity
    
    def __str__(self):
        return f"{self.name}: {self.price} USD, {self.quantity} in stock"