from sqlalchemy import create_engine, Column, Integer, String, Text, MetaData, Table
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

engine = create_engine('mysql+pymysql://root:root@localhost:3306/chat')

Base = declarative_base()


class Conversation(Base):
    __tablename__ = 'conversation'
    id = Column(Integer, primary_key=True)
    conversation_id = Column(String)
    role_name = Column(String)
    content = Column(Text)
    offset = Column(Integer)


Session = sessionmaker(bind=engine)


class ConversationManager:
    def __init__(self):
        self.session = Session()

    def close(self):
        self.session.close()

    @staticmethod
    def add(self, conversation_id: str, role_name: str, content: str) -> int:
        latest_offset = self.query_latest_offset(conversation_id=conversation_id)
        new_conversation = Conversation(
            conversation_id=conversation_id,
            role_name=role_name,
            content=content,
            offset=latest_offset
        )
        self.session.add(new_conversation)
        self.session.commit()
        return new_conversation.id

    @staticmethod
    def query_all(self):
        conversations = self.session.query(
            Conversation
        ).all()
        return conversations

    @staticmethod
    def query_latest_conversations(self, conversation_id: str) -> int:
        conversation = self.session.query(
            Conversation
        ).filter_by(
            conversation_id=conversation_id
        ).order_by(
            Conversation.offset
        ).all()[-1]
        return conversation

    @staticmethod
    def query_latest_id(self, conversation_id: str) -> int:
        conversations = self.session.query(
            Conversation
        ).filter_by(
            conversation_id=conversation_id
        ).order_by(
            Conversation.offset
        ).all()[-1]
        return conversations.id

    @staticmethod
    def query_latest_offset(self, conversation_id: str) -> int:
        conversations = self.session.query(
            Conversation
        ).filter_by(
            conversation_id=conversation_id
        ).order_by(
            Conversation.offset
        ).all()[-1]
        return conversations.offset

    @staticmethod
    def query_by_id(self, conversation_id: str) -> list[Conversation]:
        conversation = self.session.query(
            Conversation
        ).filter_by(
            conversation_id=conversation_id
        ).order_by(
            Conversation.offset
        ).all()
        return conversation

    @staticmethod
    def query_by_idWithSort(self, conversation_id: str) -> list[Conversation]:
        conversations = self.session.query(
            Conversation
        ).filter_by(
            conversation_id=conversation_id
        ).order_by(
            Conversation.offset
        ).all()
        return conversations

    @staticmethod
    def delete_by_id(self, conversation_id: str) -> bool:
        conversation = self.session.query(
            Conversation
        ).get(
            conversation_id
        )
        if conversation:
            self.session.delete(conversation)
            self.session.commit()
            return True
        return False
