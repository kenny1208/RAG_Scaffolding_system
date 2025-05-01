# -*- coding: big5 -*-

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.vectorstores import VectorStoreRetriever 

# Import models if needed for context (usually not directly needed here)
# from models import ...

def format_docs(docs):
    """Helper function to format retrieved documents."""
    return "\n\n".join([d.page_content for d in docs])

# --- Chain Creation Functions ---

def create_learning_style_survey(chat_model: ChatGoogleGenerativeAI):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""�z�O�@��M���ǲ߭���������Ш|�M�a�C
        �г]�p�@��²������Ī��ǲ߭�������ݨ��A�]�t 5 �Ӧh���D�C
        �C�Ӱ��D���� 3 �ӿﶵ�A�Ω�P�_�ǥͬO�_�D�n�O�G
        1. ��ı���ǲߪ�
        2. ťı���ǲߪ�
        3. ��ı���ǲߪ�

        �бN�z���^���榡�Ƭ��@���ݨ��A�]�t�s�������D�M�r���аO���ﶵ�C"""),
        HumanMessagePromptTemplate.from_template("�]�p�@���ǲ߭�������ݨ��C")
    ])
    return prompt | chat_model | StrOutputParser()

def create_pretest_generator(chat_model: ChatGoogleGenerativeAI, retriever: VectorStoreRetriever):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""�z�O�@��M���Ш|�����]�p���M�a�C
        �ھڴ��Ѫ����e�A�]�p�@���e���]Pre-Test�^�A�H�����ǥͦb�ӥD�D�W���{�����Ѥ����C

        �г]�p�[�\���P���ׯŧO�����D�G²��B�����M�x���C
        ���C�Ӱ��D�A�д��ѡG
        1. ���D�奻
        2. �|�ӳ��ﶵ�]A, B, C, D�^
        3. ���T����
        4. �����򥿽T������
        5. ���ׯŧO

        �z������`�H�U��T�� JSON �榡�G
        {
          "title": "�e���G[�D�D]",
          "description": "������N�����z��[�D�D]���{������",
          "questions": [
            {
              "question": "���D�奻�H",
              "choices": ["A. �ﶵ A", "B. �ﶵ B", "C. �ﶵ C", "D. �ﶵ D"],
              "correct_answer": "A. �ﶵ A",
              "explanation": "������ A �O���T���ת�����",
              "difficulty": "²��"
            }
          ]
        }

        �Юھڴ��Ѫ����e�ͦ��`�@ 5 �Ӱ��D�A�å]�t���P���ׯŧO�����D�C
        """),
        HumanMessagePromptTemplate.from_template("""�ھڥH�U���e�ͦ��@���e���G

        {context}
        """)
    ])

    pretest_chain = (
        {
            # Pass the input directly to the retriever, then format
            "context": RunnableLambda(lambda inputs: inputs.get("topic", "general knowledge")) | retriever | format_docs
        }
        | prompt
        | chat_model
        | JsonOutputParser() 
    )
    return pretest_chain


def create_learning_path_generator(chat_model: ChatGoogleGenerativeAI, retriever: VectorStoreRetriever):
    prompt = ChatPromptTemplate.from_messages([
         SystemMessage(content="""�z�O�@��M���ӤH�ƾǲ߸��|�]�p���Ш|�ҵ{�]�p�M�a�C
        �ھڴ��Ѫ��ǥ��ɮסB���絲�G�M���e�A�Ыؤ@���A�X�L�۾Ǫ��ǲ߸��|�C

        �z���ǲ߸��|���ӡG
        1. �w��ǥͪ��ǲ߭���B���Ѥ����M����i��q���w��
        2. �]�t�M�����ǲߥؼ�
        3. ��`�N�[��h�A�v�B�W�[���רô�֤��

        �z���^��������`�H�U��T�� JSON �榡�G
        {
          "title": "�w��[�D�D]���ӤH�ƾǲ߸��|",
          "description": "���ǲ߸��|�w��[name]���ǲ߭���M��e���Ѥ����i��q���w��",
          "objectives": ["�ؼ� 1", "�ؼ� 2", "�ؼ� 3"],
          "modules": [
            {
              "title": "���` 1: [���D]",
              "description": "���`�y�z",
              "activities": [
                {
                  "type": "�\Ū",
                  "title": "���ʼ��D",
                  "description": "���ʴy�z",
                  "difficulty": "��Ǫ�"
                }
              ],
              "resources": ["���q���`1-1", "���q���`1-2"],
            }
          ]
        }
        """),
        HumanMessagePromptTemplate.from_template("""�ھڥH�U���e�ͦ��ӤH�ƾǲ߸��|�G

        �ǥ��ɮסG
        {profile}

        ���絲�G�G
        {test_results}

        �������e�G
        {context}
        """)
    ])

    learning_path_chain = (
        {
            "profile": RunnablePassthrough(), # Pass profile dict directly
            "test_results": RunnablePassthrough(), # Pass results dict directly
            # Retrieve context based on a general topic or derived from profile/results if needed
            "context": RunnableLambda(lambda inputs: inputs.get("topic", "relevant subject matter")) | retriever | format_docs
        }
        | prompt
        | chat_model
        | JsonOutputParser() # Assumes LearningPath model or dict output
    )
    return learning_path_chain

def create_peer_discussion_ai(chat_model: ChatGoogleGenerativeAI, retriever: VectorStoreRetriever):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""�z�O�u�ǲ߹٦�v�A�@�Ӥ͵��B�����U�� AI �P���A�P�ǥͶi�榳�س]�ʪ��Q�סC
        �z������O�G
        1. �����@��]�b�ǲ߸ӥD�D�����@�w���Ѫ��P��
        2. ���X�P�i��P�ʫ�Ҫ��`����{�����D
        3. ���ѷũM�����ɡA�Ӥ��O�������X����
        4. �H��ܪ��覡��F�Q�k�A���O�ǥͤ�������y
        5. �ϥ�Ĭ��ԩ������ݪk���U�ǥ͵o�{����
        6. ���y�ëO���n�����A��

        �ھڴ��Ѫ��������e�^���A�����n�u�O²��a�I�w��T�C
        �ӬO�H�۵M���覡�i��Ӧ^�Q�סA�N���@�_�ǲߤ@�ˡC
        """),
        HumanMessagePromptTemplate.from_template("""�ǥͧƱ�Q�׳o�ӥD�D�G

        �D�D: {topic}

        �������e:
        {context}

        �ǥͰT��:
        {message}
        """)
    ])

    discussion_chain = (
        {
            "topic": RunnablePassthrough(), # Pass topic string directly
            "message": RunnablePassthrough(), # Pass message string directly
            "context": RunnableLambda(lambda inputs: inputs["topic"]) | retriever | format_docs
        }
        | prompt
        | chat_model
        | StrOutputParser()
    )
    return discussion_chain

def create_posttest_generator(chat_model: ChatGoogleGenerativeAI, retriever: VectorStoreRetriever):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""�z�O�@��M�~���Ш|�����]�p�v�C
        �ھڴ��Ѫ��ǲ߳��`���e�M�ǥͪ���e���Ѥ����A�]�p�@������A�]�t�h���D�H�����ǥͪ��ǲߦ��G�C

        �������P�ǥͪ���e�����۲šG
        - ��Ǫ̡G��h²����D�]70%�^�A�@�Ǥ������D�]30%�^
        - ���Ū̡G�@��²����D�]30%�^�A�D�n�O�������D�]50%�^�A�@�ǧx�����D�]20%�^
        - ���Ū̡G�@�Ǥ������D�]30%�^�A�D�n�O�x�����D�]70%�^

        �]�p�����D�����վǥ͹鷺�e���z�ѡB���ΩM���R��O�C

        ���C�Ӱ��D�A�д��ѡG
        1. ���D�奻
        2. �|�Ӧh��ﶵ�]A, B, C, D�^
        3. ���T����
        4. �����򥿽T������
        5. ���ׯŧO

        �z������`�H�U��T�� JSON �榡�G
        {
          "title": "����G[�D�D]",
          "description": "������N�����z��[�D�D]���ǲߦ��G",
          "questions": [
            {
              "question": "���D�奻�H",
              "choices": ["A. �ﶵ A", "B. �ﶵ B", "C. �ﶵ C", "D. �ﶵ D"],
              "correct_answer": "A. �ﶵ A",
              "explanation": "������ A �O���T���ת�����",
              "difficulty": "²��"
            }
          ]
        }

        �ھھǥͪ������ͦ��`�@ 5 �Ӱ��D�A�þA����t���סC
        """),
        HumanMessagePromptTemplate.from_template("""�ھڥH�U���e�ͦ��@������G

        �ǥͪ���e���Ѥ���: {knowledge_level}
        ���`���e�D�D: {module_topic}
        �������e:
        {context}
        """)
    ])

    posttest_chain = (
        {
            "knowledge_level": RunnablePassthrough(), # Pass level string directly
            "module_topic": RunnablePassthrough(), # Pass topic string directly
            "context": RunnableLambda(lambda inputs: inputs["module_topic"]) | retriever | format_docs
        }
        | prompt
        | chat_model
        | JsonOutputParser() # Assumes Test model or dict output
    )
    return posttest_chain

def create_learning_log_prompter(chat_model: ChatGoogleGenerativeAI):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""�z�O�@��ϫ�ǲ߱нm�A�M�����U�ǥͳЫئ��N�q���ǲߤ�x�C
        �ھھǥͧ������ǲ߳��`�M���絲�G�A�޾ɥL�̤ϫ�ۤv���ǲߡC

        ���X�`����{���}�񦡰��D�H�P�i�ϫ�A�]�A�G
        1. �L�̾Ǩ�F����]���䷧���M���ѡ^
        2. �L�̹�ǲ߹L�{���P��
        3. �L��ı�o���D�Ԫ��a��
        4. �L�̤��M��������D

        �z���ؼЬO���U�ǥͳЫؤ@���״I�B���ϫ�ʪ��ǲߤ�x�A��L�̪����������ȡC
        """),
        HumanMessagePromptTemplate.from_template("""���U�ǥͰ��H�U���e�Ыؾǲߤ�x�ϫ�G

        ���������`: {module_title}

        ���`���e�K�n: {module_summary}

        ���絲�G: {test_results}
        """)
    ])
    return prompt | chat_model | StrOutputParser()

def create_learning_log_analyzer(chat_model: ChatGoogleGenerativeAI):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""�z�O�@��M�~���Ш|���R�v�A�M�����R�ǥͪ��ǲߤ�x�C
        �ھھǥͪ��ǲߤ�x�A�����G

        1. �����䷧�����z�ѵ{��
        2. �u�թM�۫H�����
        3. �V�c�λ~�Ѫ����
        4. ����ƪ����P����
        5. �ǲ߭��檺����

        �N�z���^���榡�Ƭ��H�U��T�� JSON ���c:
        {
          "understanding_level": "��/��/�C",
          "strengths": ["�u�� 1", "�u�� 2"],
          "areas_for_improvement": ["��i��� 1", "��i��� 2"],
          "emotional_response": "�ﱡ�P�������y�z",
          "learning_style_indicators": ["���� 1", "���� 2"],
          "recommended_next_steps": ["��ĳ�B�J 1", "��ĳ�B�J 2"],
          "suggested_resources": ["�귽 1", "�귽 2"]
        }
        """),
        HumanMessagePromptTemplate.from_template("""���R�H�U�ǲߤ�x�G

        �ǥ�: {student_name}
        �D�D: {topic}
        �ǲߤ�x���e:
        {log_content}
        """)
    ])
    return prompt | chat_model | JsonOutputParser() # Assumes dict output

def create_knowledge_level_assessor(chat_model: ChatGoogleGenerativeAI):
    # Note: Original code had StrOutputParser but the prompt asks for JSON.
    # Assuming JSON output is desired based on the prompt's format specification.
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""�z�O�@��Ш|�����M�a�C
        �ھھǥͪ����絲�G�A�T�w�L�̦b���S�w�D�D�W�����Ѥ����C

        �Ҽ{�G
        1. ���T���ת��ƶq
        2. ���T�^�������D����
        3. ���׼Ҧ��]�@�P���z�ѻP�t�Z�^

        �N�ǥͪ����Ѥ����������G
        - ��Ǫ̡G�򥻼��x�A�z��²�淧��
        - ���Ū̡G�}�n���֤߷����z�ѡA�@�w�����ί�O
        - ���Ū̡G�`��z�ѡA��N�������Ω�s����

        ���z����������²�u���z�ѡC

        �N�z���^���榡�Ƭ� JSON ��H�G
        {
          "knowledge_level": "��Ǫ�/���Ū�/���Ū�",
          "justification": "�������²�u����",
          "strengths": ["�u�� 1", "�u�� 2"],
          "areas_for_improvement": ["��i��� 1", "��i��� 2"],
          "recommended_focus": "�ǥͱ��U�����ӱM�`�󤰻�"
        }
        """),
        HumanMessagePromptTemplate.from_template("""�ھڥH�U���絲�G�����ǥͪ����Ѥ����G

        ����: {test_title}

        ���D�M����:
        {test_results_details}
        """) # Changed key name for clarity
    ])
    return prompt | chat_model | JsonOutputParser() # Changed to JsonOutputParser

def create_module_content_generator(chat_model: ChatGoogleGenerativeAI, retriever: VectorStoreRetriever):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""�z�O�@��M�~���Ш|���e�Ч@�̡C
        �ھڴ��Ѫ����`�D�D�H�ξǥͪ��ǲ߭���M���Ѥ����A�ЫؤޤH�J�Ӫ��Ш|���e�C

        �z�����e���G
        1. �w��ǥͪ��ǲ߭���]��ı���Bťı���ΰ�ı���^�i��q���w��
        2. �����ŦX�N�[�Ш|�z��
        3. �A�X�ǥͪ����Ѥ���
        4. �]�t���䷧�����M������
        5. ���c�M���A�]�t���T�������M���D
        6. �H�����I��²�u�`������
        7. ���n�ϥιϪ�ιϹ��A�i�H�ϥΪ��

        �ϥ� markdown �榡�Ʊz�����e�H�����iŪ�ʡC
        """),
        HumanMessagePromptTemplate.from_template("""���H�U���e�ЫرШ|���e�G

        ���`�D�D: {module_topic}
        �ǥ;ǲ߭���: {learning_style}
        �ǥͪ��Ѥ���: {knowledge_level}

        �����ӷ�����:
        {context}
        """)
    ])

    content_chain = (
        {
            "module_topic": RunnablePassthrough(), # Pass topic string
            "learning_style": RunnablePassthrough(), # Pass style string
            "knowledge_level": RunnablePassthrough(), # Pass level string
            "context": RunnableLambda(lambda inputs: inputs["module_topic"]) | retriever | format_docs
        }
        | prompt
        | chat_model
        | StrOutputParser()
    )
    return content_chain