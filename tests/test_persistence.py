# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
import sys, os, tempfile
sys.path.insert(0, ".")

from agent_bmm.persistence import ConversationStore


def test_conversation_lifecycle():
    db = os.path.join(tempfile.mkdtemp(), "test.db")
    store = ConversationStore(db)

    # Create
    cid = store.create_conversation("Test chat")
    assert cid > 0

    # Add messages
    store.add_message(cid, "user", "Hello")
    store.add_message(cid, "assistant", "Hi!", tool_name="greeting", expert_id=0)
    store.add_message(cid, "user", "How are you?")

    # Get
    msgs = store.get_messages(cid)
    assert len(msgs) == 3
    assert msgs[0]["role"] == "user"
    assert msgs[1]["tool_name"] == "greeting"
    print("OK: conversation lifecycle")


def test_search():
    db = os.path.join(tempfile.mkdtemp(), "test.db")
    store = ConversationStore(db)
    cid = store.create_conversation("Search test")
    store.add_message(cid, "user", "Tell me about quantum physics")
    store.add_message(cid, "assistant", "Quantum physics studies particles")

    results = store.search_messages("quantum")
    assert len(results) == 2
    print("OK: search")


def test_agent_state():
    db = os.path.join(tempfile.mkdtemp(), "test.db")
    store = ConversationStore(db)
    cid = store.create_conversation("State test")
    store.save_agent_state(cid, step=1, routing_decisions=[0, 1, 2], expert_distribution={"search": 1, "math": 2})
    print("OK: agent state")


def test_list_and_delete():
    db = os.path.join(tempfile.mkdtemp(), "test.db")
    store = ConversationStore(db)
    c1 = store.create_conversation("First")
    c2 = store.create_conversation("Second")

    convs = store.get_conversations()
    assert len(convs) == 2

    store.delete_conversation(c1)
    convs = store.get_conversations()
    assert len(convs) == 1
    assert convs[0]["title"] == "Second"
    print("OK: list and delete")


if __name__ == "__main__":
    test_conversation_lifecycle()
    test_search()
    test_agent_state()
    test_list_and_delete()
    print("\nAll persistence tests passed!")
