# 1. 도서관 관리 시스템
#       요구사항:
#           Book 클래스를 생성하고, 제목, 저자, 책번호을 속성으로 추가.
#           Library 클래스를 생성하여 책 목록을 관리하며, 책을 추가/삭제하는 메서드 구현.
#           사용자가 원하는 제목으로 책을 검색할 수 있는 메서드 구현.

#제목- Title/ 저자 Author/ 책번호 BNum

class Book:
    def __init__(self, title, author, bnum):
        self.title = title
        self.author = author
        self.bnum = bnum

class Library:
    def __init__(self):
        self.books = []

    def add_book(self, book):
        self.books.append(book)
        print(f"책 '{book.title}'이 추가되었습니다.")

    def remove_book(self, bnum):
        for book in self.books:
            if book.bnum == bnum:
                self.books.remove(book)
                print(f"책 '{book.title}'이 삭제되었습니다.")
                return
        print("해당 ISBN의 책이 없습니다.")

    def search_by_title(self, title):
        results = [book for book in self.books if title.lower() in book.title.lower()]
        if results:
            print("검색 결과:")
            for book in results:
                print(f"- {book.title} by {book.author} (ISBN: {book.bnum})")
        else:
            print("검색 결과가 없습니다.")

# 테스트 코드
library = Library()
book1 = Book("파이썬 입문", "홍길동", "12345")
book2 = Book("파이썬 심화", "이몽룡", "67890")
book3 = Book("자바스크립트 기초", "성춘향", "54321")

library.add_book(book1)
library.add_book(book2)
library.add_book(book3)

library.search_by_title("파이썬")
library.remove_book("12345")
library.search_by_title("파이썬")


        
        
    

    