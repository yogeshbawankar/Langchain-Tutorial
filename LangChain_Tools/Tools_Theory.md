---
title: 'Building Agents with LangChain: The Role of Tools'
description: An overview of how Tools in LangChain overcome the limitations of LLMs, enabling them to perform real-world actions and form the basis of AI agents.
author: Yogesh Bawankar
date: 2025-08-24
tags: [LangChain, LLM, AI Agents, Tools, LangGraph, Python]
slug: langchain-agents-tools-guide
---

# Building Agents with LangChain: The Role of Tools

*This guide explores the foundational concept of Tools in LangChain, explaining how they empower Large Language Models (LLMs) to execute external tasks and act as the core building block for creating capable AI agents.*

---

## 📜 Table of Contents

- [The Foundational Role of Tools](#-the-foundational-role-of-tools)
- [Understanding Tools in LangChain](#-understanding-tools-in-langchain)
  - [Tool Categories: Built-in vs. Custom](#tool-categories-built-in-vs-custom)
  - [How LLMs Interact with Tools](#how-llms-interact-with-tools)
- [The Synergy Between Tools and Agents](#-the-synergy-between-tools-and-agents)
- [How to Develop Custom Tools](#-how-to-develop-custom-tools)
  - [1. Using the `@tool` Decorator (Simple)](#1-using-the-tool-decorator-simple)
  - [2. Using `StructuredTool` with Pydantic (Strict)](#2-using-structuredtool-with-pydantic-strict)
  - [3. Inheriting from the `BaseTool` Class (Advanced)](#3-inheriting-from-the-basetool-class-advanced)
- [Organizing with Toolkits](#-organizing-with-toolkits)
- [Key Terminology](#-key-terminology)
- [Key Takeaways](#-key-takeaways)

---

## 🧠 The Foundational Role of Tools

Large Language Models (LLMs) are powerful systems with two core capabilities: **reasoning** and **language generation**. They can think through a problem to determine how to answer it and then speak by generating a word-by-word response.

However, LLMs inherently lack the ability to perform external actions. They are like a brain and mouth without hands or legs—they can think and talk but cannot execute tasks in the real world. For instance, an LLM can't fetch live weather data, reliably solve complex math problems, call an external API, run code, or interact with a database on its own.

**Tools** bridge this gap. They serve as the "hands and legs" for an LLM, connecting it to external functions or APIs and enabling it to perform any given task. For example, while an LLM can suggest ways to travel from Delhi to Mumbai, it cannot book a train ticket without a specific tool designed for that purpose. The more tools an LLM has access to, the more capable and versatile it becomes.

---

## 🛠️ Understanding Tools in LangChain

In LangChain, a **Tool** is essentially a Python function packaged in a way that an LLM can understand and invoke when needed. The LLM autonomously decides which tool to use, provides the necessary inputs, executes the tool, and processes the returned results to complete its objective.

### Tool Categories: Built-in vs. Custom

LangChain offers two primary categories of tools:

1.  **Built-in Tools**: These are pre-built by the LangChain team for common and popular use cases. They are production-ready and require minimal setup, allowing developers to quickly integrate functionalities like web searches, Wikipedia queries, or running shell commands.
2.  **Custom Tools**: These are user-created tools designed for specific use cases not covered by the built-in options. They are necessary for tasks like calling a company's proprietary APIs, encapsulating unique business logic, or enabling the LLM to interact with custom applications.

Here are a few popular **built-in tools**:

| Tool Name                 | Description                             | Example Use Case                                   |
| ------------------------- | --------------------------------------- | -------------------------------------------------- |
| `DuckDuckGoSearchRun`     | Performs web searches.                  | Find "Top news in India today".                    |
| `WikipediaQueryRun`       | Searches for articles on Wikipedia.     | Get information about the Eiffel Tower.            |
| `PythonREPLTool`          | Executes raw Python code.               | Calculate the factorial of a number.               |
| `ShellTool`               | Runs shell commands.                    | List files in the current directory with `ls`.     |
| `RequestsGetTool`         | Makes HTTP GET requests.                | Fetch data from a public API endpoint.             |
| `GmailSendMessageTool`    | Sends emails via a Gmail account.       | Email a summary of a task to a colleague.          |
| `SQLDatabaseQueryTool`    | Executes SQL queries against a database.| Retrieve customer data from a sales database.      |

### How LLMs Interact with Tools

When a tool is provided to an LLM, the model doesn't see the underlying Python logic. Instead, it receives a JSON schema describing the tool's **name**, **description**, and **arguments**. This schema is crucial as it tells the LLM what the tool does and how to use it, allowing it to make informed decisions about when to call it.

---

## 🤖 The Synergy Between Tools and Agents

An **AI Agent** is an LLM-powered system that can autonomously think, make decisions, and take actions to achieve a goal. This functionality is a direct result of combining an LLM with tools.

Agents possess two core capabilities that mirror this synergy:
* **Reasoning and Decision Making**: The ability to think step-by-step to solve a problem, a capability derived from the LLM.
* **Action Performance**: The ability to execute tasks once a decision is made, a capability powered by **Tools**.

Essentially, the combination of an **LLM (the brain) + Tools (the hands)** is what constitutes an agent. Therefore, understanding tools is as crucial as understanding LLMs for building effective agents in LangChain or LangGraph.

---

## 🔧 How to Develop Custom Tools

You'll need custom tools when you want to call your own APIs, encapsulate unique business logic, or let an LLM interact with your private databases and applications. LangChain provides three primary methods for creating them.

### 1. Using the `@tool` Decorator (Simple)

This is the simplest and most common method for creating a custom tool.

* **Process**:
    1.  Define a standard Python function (e.g., `multiply_two_numbers`).
    2.  Add a **docstring** to describe what the function does. This becomes the tool's description for the LLM.
    3.  Use **type hinting** for inputs and outputs to help the LLM understand data types.
    4.  Apply the `@tool` decorator directly above the function definition.
* **Characteristics**: This method automatically converts your Python function into a runnable LangChain tool.

### 2. Using `StructuredTool` with Pydantic (Strict)

This approach offers more explicit control over the tool's input schema, making it ideal for production-ready agents.

* **Process**:
    1.  Define a Pydantic `BaseModel` class to specify the tool's input schema, including field names, types, and descriptions.
    2.  Create a standard Python function that contains the tool's logic.
    3.  Instantiate the tool using `StructuredTool.from_function()`, providing the function, a name, a description, and the Pydantic model as the `args_schema`.
* **Characteristics**: Provides stricter enforcement of input constraints, reducing errors.

### 3. Inheriting from the `BaseTool` Class (Advanced)

This method provides the deepest level of customization and is necessary for advanced use cases like asynchronous operations.

* **Process**:
    1.  Create a custom class that inherits from LangChain's `BaseTool`.
    2.  Define class attributes for `name`, `description`, and `args_schema` (using a Pydantic model).
    3.  Implement the `_run` method for the tool's synchronous execution logic.
    4.  Optionally, implement the `_arun` method for asynchronous execution.
* **Characteristics**: Offers maximum flexibility, including support for concurrency, which is not available with the other methods.

---

## 🧰 Organizing with Toolkits

A **Toolkit** is a collection of related tools grouped together to serve a common purpose. For example, if you create several tools for interacting with Google Drive (e.g., upload, search, read file), you can package them into a single `GoogleDriveToolkit`.

This approach offers two main benefits:
* **Convenience**: Simplifies the management of related tools.
* **Reusability**: Allows the entire toolkit to be easily reused across different agents and applications.

To create a custom toolkit, define a class that contains a `get_tools()` method, which returns a list of all the tool objects in that toolkit.

---

## 📚 Key Terminology

* **LLMs (Large Language Models)**: AI systems with advanced reasoning and language generation capabilities but no inherent ability to perform external actions.
* **Tools**: Python functions that give LLMs "hands and legs" to interact with the real world by searching, running code, or calling APIs.
* **Built-in Tools**: Pre-made, ready-to-use tools provided by LangChain for common tasks.
* **Custom Tools**: Tools created by developers for unique use cases, such as calling proprietary APIs or interacting with private databases.
* **`@tool` Decorator**: A simple way to convert a Python function into a LangChain tool.
* **Docstring**: A string in a Python function that serves as the tool's description for the LLM.
* **Type Hinting**: Code annotations that specify expected data types, helping the LLM use the tool correctly.
* **`StructuredTool`**: A tool type that uses a Pydantic model to define a strict input schema.
* **Pydantic Model**: A data validation library used to define and enforce structured schemas for tool inputs.
* **`BaseTool` Class**: The abstract base class from which all LangChain tools inherit, allowing for deep customization and asynchronous operations.
* **`_run` Method**: The method that defines a tool's synchronous execution logic when using the `BaseTool` class.
* **`_arun` Method**: The method that defines a tool's asynchronous execution logic.
* **Toolkits**: Collections of related tools grouped together for convenience and reusability.
* **Agents (AI Agents)**: LLM-powered systems that autonomously use tools to think, make decisions, and perform actions to achieve a goal.

---

## ✨ Key Takeaways

* Integrate **tools** to extend LLM capabilities beyond language generation, enabling them to perform real-world tasks like web searches or API calls.
* Leverage LangChain's **built-in tools** for common functionalities to accelerate development.
* Create **custom tools** when built-in options don't meet your specific needs, such as interacting with proprietary systems.
* Always include a clear **docstring** and use **type hinting** in custom tools; this is critical for the LLM to understand and use the tool correctly.
* For production applications requiring strict input validation, use `StructuredTool` with **Pydantic models**.
* For advanced customization or asynchronous operations, build tools by inheriting from the `BaseTool` class.
* Group related tools into **Toolkits** to improve organization and reusability.

> ⚠️ **Warning**: Exercise caution when using powerful tools like the `ShellTool` in production environments. They can execute arbitrary commands that could lead to unintended system modifications or security vulnerabilities.