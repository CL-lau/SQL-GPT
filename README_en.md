# SQL_GPT: Automated SQL Query Generation Tool

## Introduction

SQL_GPT is a powerful tool that can automatically generate SQL query statements based on simple textual descriptions. Whether you're quickly generating complex queries or interacting with databases, SQL_GPT can significantly enhance your efficiency and workflow.

## Features and Capabilities

### Automated SQL Query Generation

SQL_GPT can intelligently generate SQL query statements that meet your requirements based on textual descriptions provided by the user. This eliminates the need for an in-depth understanding of SQL syntax; you can simply describe your query needs in natural language.

### Error Correction and Suggestions

If there are issues with the generated SQL queries, worry not! SQL_GPT will analyze error messages and provide suggestions for corrections, helping you swiftly resolve problems and ensuring the accuracy of your query statements.

### Database Connection and Interaction

SQL_GPT supports configuring real database connection information, enabling you to execute the generated SQL queries directly within the tool and interact with databases in real-time. No need to switch environments; everything is at your fingertips.

### Multi-Database Support

Whether you're using MySQL, PostgreSQL, or Microsoft SQL Server, SQL_GPT is compatible with various mainstream database systems. This provides greater flexibility for different project requirements.

### User-Friendly Interface

SQL_GPT offers an intuitive user interface that makes it easy for both beginners and experienced professionals to get started. The straightforward and intuitive interface enhances your productivity.

## Feature List

- [x] **Automated SQL Query Generation:** Generate SQL query statements effortlessly by describing your needs in simple text.
- [x] **Error Correction Suggestions:** Receive intelligent suggestions for correcting queries with errors, aiding in quick issue resolution.
- [x] **Database Connection Management:** Easily configure and manage multiple database connections, execute generated SQL queries within the tool.
- [x] **Multi-Database Compatibility:** Compatible with various mainstream database systems, catering to different project needs.
- [x] **Proxy Access Support:** For specific scenarios, you can access the GPT service through system proxies.
- [ ] **Multi API KEY Rotation:** Enhance stability by setting up multiple alternative API keys to access GPT.

## Quick Start Guide

To begin using SQL_GPT, follow these simple steps:

1. **Install Required Dependencies:** Ensure you have Python 3.x installed in your environment and execute the following command to install the required dependencies:

    ```bash
    pip install requirements.txt
    ```

2. **Configure Database Connection:** Configure your database connection information in the tool, including hostname, username, password, etc., for seamless database interaction.

3. **Generate SQL Queries:** Within the tool's user interface, use natural language to describe your query requirements. SQL_GPT will intelligently generate the corresponding SQL query statements.

## Usage Example

**Input Description:** I want to find users who are older than 25 years.

**Generated SQL Query:**
```sql
SELECT * FROM users WHERE age > 25;
```

## Star history

[![Star History Chart](https://api.star-history.com/svg?repos=CL-lau/SQL-GPT&type=Date)](https://star-history.com/#CL-lau/SQL-GPT&Date)