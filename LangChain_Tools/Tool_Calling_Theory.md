---
title: "Mastering Tool Calling in LangChain: A Step Towards AI Agents"
description: "An exploration of Tool Calling in LangChain, detailing the workflow from creation and binding to execution, and demonstrating a practical currency conversion application with sequential tool calls."
author: "Yogesh Bawankar"
tags: [LangChain, AI, LLM, Tool Calling, AI Agents, Python]
date: 2025-08-24
slug: langchain-tool-calling-guide
---

# Mastering Tool Calling in LangChain: A Step Towards AI Agents

*This guide explores the concept of **Tool Calling** in LangChain, a fundamental step toward building sophisticated AI agents. We'll break down the entire workflow, from creating and binding tools to executing them, and build a practical real-time currency conversion application to handle sequential tool calls.*

---

## Table of Contents

- [Key Themes & Arguments](#key-themes--arguments-)
- [Understanding LLM Capabilities and Limitations](#understanding-llm-capabilities-and-limitations-)
  - [LLM Strengths](#llm-strengths)
  - [LLM Limitations](#llm-limitations)
- [Introduction to Tools](#introduction-to-tools-)
  - [Purpose and Nature of Tools](#purpose-and-nature-of-tools)
  - [Core Tool Components](#core-tool-components)
- [The Tool Binding Process](#the-tool-binding-process-)
  - [How Binding Empowers the LLM](#how-binding-empowers-the-llm)
  - [Practical Implementation Example](#practical-implementation-example)
- [The Tool Calling and Execution Workflow](#the-tool-calling-and-execution-workflow-)
  - [How an LLM Decides to Call a Tool](#how-an-llm-decides-to-call-a-tool)
  - [The `tool_calls` Output](#the-tool_calls-output)
  - [Critical Distinction: LLMs Suggest, Programmers Execute](#critical-distinction-llms-suggest-programmers-execute)
  - [Executing the Tool and Using `ToolMessage`](#executing-the-tool-and-using-toolmessage)
  - [Maintaining Conversation History](#maintaining-conversation-history)
- [Building a Practical Application: Real-time Currency Conversion](#building-a-practical-application-real-time-currency-conversion-)
  - [The Challenge: Outdated LLM Knowledge](#the-challenge-outdated-llm-knowledge)
  - [The Solution: A Two-Tool Approach](#the-solution-a-two-tool-approach)
  - [The Sequential Dependency Problem](#the-sequential-dependency-problem)
  - [The Fix: `InjectedToolArgument`](#the-fix-injectedtoolargument)
  - [Final Sequential Execution Flow](#final-sequential-execution-flow)
- [Conclusion: Tool-Using Applications vs. True AI Agents](#conclusion-tool-using-applications-vs-true-ai-agents-)
- [Key Terminology](#key-terminology-)
- [Key Takeaways](#key-takeaways-)

---

## Key Themes & Arguments 🔑

- **LLM Capabilities and Limitations**: Understanding what Large Language Models (LLMs) can and cannot do independently.
- **Function of Tools**: Exploring how tools extend LLM capabilities to perform real-world actions.
- **Tool Binding Process**: Connecting tools with LLMs to inform them about available functionalities.
- **Tool Calling Workflow**: The step-by-step process from an LLM's decision to use a tool to the integration of its output.
- **Practical Application**: Building a real-world use case for sequential tool calls and handling dependencies.

---

## Understanding LLM Capabilities and Limitations 🤔

### LLM Strengths

- **Reasoning**: LLMs excel at breaking down and understanding complex questions.
- **Output Generation**: They access their vast parametric knowledge to generate relevant, text-based answers.
- **Analogy**: An LLM is like a person who is excellent at thinking and speaking.

### LLM Limitations

- **Inability to Act**: LLMs cannot directly perform real-world tasks. They can't modify a database, post on social media, or hit an API to fetch real-time data like weather or currency rates.
- **Analogy**: They are like a person who can think and speak but lacks the "hands and feet" to carry out actions in the world.
- **The Core Problem**: This limitation is a significant hurdle in making LLMs powerful entities in the software ecosystem.

---

## Introduction to Tools 🛠️

### Purpose and Nature of Tools

The solution to LLM limitations is to create explicit **Tools**. Each tool is a specialized function designed to carry out a specific task, effectively giving the LLM the "hands and feet" it lacks.

Examples of tools include:
- **DuckDuckGo Search**: To search the web.
- **Shell Tool**: To run command-line commands.

In essence, tools are special Python functions capable of interacting with an LLM when required.

### Core Tool Components

As covered previously, every tool generally consists of:
- **Name**: A unique identifier.
- **Description**: An explanation of what the tool does, which allows the LLM to understand its function.
- **Input Schema**: A definition of the expected format for the tool's inputs.

---

## The Tool Binding Process 🔗

**Tool Binding** is the process of connecting or registering tools with a Large Language Model.

### How Binding Empowers the LLM

- **Awareness**: The LLM knows which tools are available for it to use.
- **Understanding**: It comprehends the function of each tool through its description.
- **Correct Usage**: It knows the correct input format (Input Schema) required to invoke each tool, ensuring proper communication.

### Practical Implementation Example

1.  **Tool Creation**: A `multiply` tool is created using a decorator. It expects two integer inputs (`A`, `B`), returns an integer, and has a clear docstring.
2.  **LLM Initialization**: An LLM instance, such as `ChatOpenAI`, is created.
3.  **Binding**: The `bind_tools()` function of the LLM is called, passing a list of tools (e.g., `[multiply]`). The result is a new, tool-aware LLM instance stored in a variable like `llm_with_tools`.

> 📌 **Note**: Not all LLMs support tool binding. This capability is specific to certain models.

---

## The Tool Calling and Execution Workflow ⚙️

### How an LLM Decides to Call a Tool

**Tool Calling** is the process where an LLM, in response to a query, decides to use a specific tool. It generates a structured output indicating the tool's name and the arguments to use.

The LLM determines if a query requires a tool based on its context.
- **No Tool Needed**: If a query like "Hi, how are you?" can be answered with parametric knowledge, no tool is called.
- **Tool Needed**: If a query like "Can you multiply 3 by 10?" requires an external action, the LLM will call the appropriate tool.

### The `tool_calls` Output

When an LLM decides to use a tool, its response will contain a `tool_calls` attribute, which is a list. Each item in this list is a dictionary specifying:
- **`name`**: The name of the suggested tool (e.g., `multiply`).
- **`args`**: A dictionary of input arguments for the tool (e.g., `{'A': 3, 'B': 10}`).
- **`id`**: A unique identifier for this specific tool call.
- **`type`**: The message type, which is `"tool_call"`.

### Critical Distinction: LLMs Suggest, Programmers Execute

> ⚠️ **Warning**: The LLM does **not** execute the tool itself. It only *suggests* which tool to use and what arguments to pass.

The actual execution is handled by the programmer or the LangChain framework. This is a crucial safety measure, as it ensures the programmer retains control and prevents LLMs from taking autonomous, and potentially harmful, actions.

### Executing the Tool and Using `ToolMessage`

**Tool Execution** is the step where the programmer runs the Python function (the tool) using the arguments suggested by the LLM.

The result of this execution is then wrapped in a special `ToolMessage`. This message is sent back to the LLM to provide it with the context of what happened and the tool's output, closing the loop.

### Maintaining Conversation History

To ensure the LLM has full context, a chronological history of the conversation is maintained in a list (e.g., `messages`). The flow is as follows:

1.  The initial user query is stored as a `HumanMessage`.
2.  The LLM's response, containing the `tool_calls`, is stored as an `AIMessage`.
3.  After the programmer executes the tool, the `ToolMessage` (containing the tool's result) is created.
4.  All three messages (`HumanMessage`, `AIMessage`, `ToolMessage`) are appended to the `messages` list.
5.  This complete history is passed back to the LLM for a final, context-aware response.

---

## Building a Practical Application: Real-time Currency Conversion 💵

### The Challenge: Outdated LLM Knowledge

LLMs have a knowledge cut-off date, meaning their information on currency conversion rates is outdated. A real-time conversion requires fetching current data from an external source.

### The Solution: A Two-Tool Approach

We can solve this by creating two tools: one to fetch the conversion factor from an API and another to perform the multiplication.

1.  **Tool 1: `get_conversion_factor`**
    - **Purpose**: Fetches the real-time conversion factor between a `base_currency` and a `target_currency` by calling an external API (e.g., Exchange Rate API).
    - **Inputs**: `base_currency` (string), `target_currency` (string).
    - **Output**: The conversion rate (float).

2.  **Tool 2: `convert`**
    - **Purpose**: Calculates the target value by multiplying a `base_currency_value` by a `conversion_rate`.
    - **Inputs**: `base_currency_value` (integer), `conversion_rate` (float).
    - **Output**: The converted value (float).

Both tools are then bound to the LLM using `bind_tools()`.

### The Sequential Dependency Problem

When a user asks to fetch the rate and then convert, a problem arises. The LLM generates two `tool_calls` simultaneously. However, the `convert` tool call requires the `conversion_rate`, which has not yet been calculated by the `get_conversion_factor` tool. The LLM attempts to fill this argument using its outdated parametric knowledge (e.g., a hardcoded rate of 73.73), which breaks the intended sequential logic.

### The Fix: `InjectedToolArgument`

The solution is to use `InjectedToolArgument`.

- **Implementation**: Modify the `conversion_rate` argument in the `convert` tool's definition using `Annotated` from Python's `typing` module: `Annotated[float, InjectedToolArgument]`.
- **Function**: This tells the LLM *not* to attempt to fill this argument during tool calling. It signals that the programmer will "inject" this value later after running the prerequisite tools.
- **Result**: The LLM's `tool_calls` for the `convert` tool will now only include the `base_currency_value`, leaving the `conversion_rate` argument empty for manual injection.

### Final Sequential Execution Flow

1.  Iterate through the `tool_calls` list from the LLM's `AIMessage`.
2.  When you encounter `get_conversion_factor`, invoke it, store the `ToolMessage`, and extract the `conversion_rate` from its output.
3.  When you encounter `convert`, fetch its current arguments.
4.  Manually add (inject) the `conversion_rate` obtained from the first tool's execution into the `convert` tool's arguments.
5.  Invoke the `convert` tool with the now-complete arguments and store its `ToolMessage`.
6.  Pass the complete message history (`HumanMessage`, `AIMessage`, `ToolMessage` #1, `ToolMessage` #2) to the LLM for a final, coherent answer.

---

## Conclusion: Tool-Using Applications vs. True AI Agents 🤖

The application we built is a powerful tool-using system, but it is **not a true AI Agent**.

- **Lack of Autonomy**: It lacks true autonomy because the programmer performed significant manual coding and orchestration, such as deciding the order of tool execution and injecting arguments.
- **What Makes a True AI Agent?**: A true **AI Agent** is autonomous. It can break down a problem and solve it step-by-step *independently*, without requiring human intervention in its decision-making or execution flow. It would dynamically decide which tool to call, in what order, and how to chain their inputs and outputs on its own.

The concepts covered here are the foundational building blocks for creating such agents, which will be explored next.

---

## Key Terminology 📚

-   **LLM (Large Language Model)**: An AI model that excels at reasoning and generating human-like text.
-   **Tools**: Specialized Python functions or utilities that enable an LLM to perform real-world tasks like searching or fetching data.
-   **Tool Binding**: The process of registering tools with an LLM to make it aware of their functions and input formats.
-   **Tool Calling**: The LLM's process of deciding to use a tool and generating a structured output with the tool's name and arguments.
-   **Tool Execution**: The actual running of a tool's function by the programmer, using the arguments suggested by the LLM.
-   **`ToolMessage`**: A special message type in LangChain that encapsulates the output of a tool execution to be sent back to the LLM.
-   **Conversation History**: A sequence of messages (`HumanMessage`, `AIMessage`, `ToolMessage`) that provides the LLM with the full context of an interaction.
-   **`InjectedToolArgument`**: A mechanism to prevent an LLM from filling a tool argument, indicating it will be supplied later by the programmer.
-   **AI Agent**: An autonomous system that can independently break down problems, make decisions, and execute multi-step tasks using tools.

---

## Key Takeaways ✅

-   **Extend LLMs with Tools**: LLMs are thinkers, not doers. Use tools to give them "hands and feet" to interact with APIs, databases, and other external services.
-   **Bind Tools Explicitly**: Always bind your tools to the LLM with clear names, descriptions, and input schemas for effective integration.
-   **You Control Execution**: Remember, the LLM only *suggests* which tools to use. The programmer is responsible for the actual execution, providing a critical layer of safety and control.
-   **Maintain Conversation History**: Pass the complete conversation history back to the LLM on each turn. This gives it the context needed to understand tool outputs and generate accurate final responses.
-   **Use `InjectedToolArgument` for Sequential Tasks**: For multi-tool workflows where one tool's output is another's input, use `InjectedToolArgument` to manage dependencies and ensure correct data flow.
-   **Distinguish Tool-Using Apps from AI Agents**: The techniques here are foundational, but a true AI agent acts with full autonomy. The orchestration we performed manually is what separates this application from a true agent.