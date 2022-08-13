# -*- coding: utf-8 -*-
# Copyright 2021 The Dapr Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import random

from bank_actor_interface import BankActorInterface
from checking_account_actor_interface import CheckingAccountActorInterface
from customer_actor_interface import CustomerActorInterface
from dapr.actor import ActorId, ActorProxy
from savings_account_actor_interface import SavingsAccountActorInterface


async def main():
    ITERATIONS = 300
    # Create proxy clients
    # Bank
    bank_1_proxy = ActorProxy.create("BankActor", ActorId("1"), BankActorInterface)
    # Customer 1
    cx_1_proxy = ActorProxy.create(
        "CustomerActor", ActorId("2"), CustomerActorInterface
    )
    cx_1_checking_acct_1_proxy = ActorProxy.create(
        "CheckingAccountActor", ActorId("3"), CheckingAccountActorInterface
    )
    cx_1_savings_acct_1_proxy = ActorProxy.create(
        "SavingsAccountActor", ActorId("4"), SavingsAccountActorInterface
    )
    # Customer 2
    cx_2_proxy = ActorProxy.create(
        "CustomerActor", ActorId("5"), CustomerActorInterface
    )
    cx_2_checking_acct_1_proxy = ActorProxy.create(
        "CheckingAccountActor", ActorId("6"), CheckingAccountActorInterface
    )
    cx_2_savings_acct_1_proxy = ActorProxy.create(
        "SavingsAccountActor", ActorId("7"), SavingsAccountActorInterface
    )

    print("BankActor.set_up", flush=True)
    await bank_1_proxy.SetUp()

    print("Initialize customer 1", flush=True)
    print("CustomerActor.set_up", flush=True)
    await cx_1_proxy.SetUp(
        {
            "checking_account": cx_1_checking_acct_1_proxy.actor_id.id,
            "savings_account": cx_1_savings_acct_1_proxy.actor_id.id,
        }
    )
    print("CheckingAccountActor.set_up", flush=True)
    await cx_1_checking_acct_1_proxy.SetUp(
        {
            "balance": 100,
            "checking_account": cx_1_checking_acct_1_proxy.actor_id.id,
            "customer": cx_1_proxy.actor_id.id,
        }
    )
    print("SavingsAccountActor.set_up", flush=True)
    await cx_1_savings_acct_1_proxy.SetUp(
        {
            "balance": 100,
            "savings_account": cx_1_savings_acct_1_proxy.actor_id.id,
            "customer": cx_1_proxy.actor_id.id,
        }
    )

    print("Initialize customer 2", flush=True)
    print("CustomerActor.set_up", flush=True)
    await cx_2_proxy.SetUp(
        {
            "checking_account": cx_2_checking_acct_1_proxy.actor_id.id,
            "savings_account": cx_2_savings_acct_1_proxy.actor_id.id,
        }
    )
    print("CheckingAccountActor.set_up", flush=True)
    await cx_2_checking_acct_1_proxy.SetUp(
        {
            "balance": 100,
            "checking_account": cx_2_checking_acct_1_proxy.actor_id.id,
            "customer": cx_2_proxy.actor_id.id,
        }
    )
    print("SavingsAccountActor.set_up", flush=True)
    await cx_2_savings_acct_1_proxy.SetUp(
        {
            "balance": 100,
            "savings_account": cx_2_savings_acct_1_proxy.actor_id.id,
            "customer": cx_2_proxy.actor_id.id,
        }
    )

    print("BankActor.run", flush=True)
    rtn_obj = await bank_1_proxy.Run()
    print(f"Return object: {rtn_obj}", flush=True)

    for i in range(1, ITERATIONS + 1):
        print(f"Iteration {i}", flush=True)
        send_amount = random.randint(10, 100)
        if (i % 2) == 0:
            print("🎲 Even", flush=True)
            print("BankActor.transfer", flush=True)
            await bank_1_proxy.Transfer(
                {
                    "amount": send_amount,
                    "from_customer": cx_2_proxy.actor_id.id,
                    "to_customer": cx_1_proxy.actor_id.id,
                }
            )
            # From withdrawal
            print("CustomerActor.withdrawal", flush=True)
            await cx_2_proxy.Withdrawal(send_amount)
            print("CheckingAccountActor.withdrawal", flush=True)
            transaction_result = await cx_2_checking_acct_1_proxy.Withdrawal(
                send_amount
            )
            if transaction_result != -1:
                # Not enough money in checking account
                print("SavingsAccountActor.withdrawal", flush=True)
                transaction_result = await cx_2_savings_acct_1_proxy.Withdrawal(
                    send_amount
                )
                if transaction_result == 1:
                    # Not enough money in savings account
                    print(
                        f"Not enough money in savings account {cx_2_savings_acct_1_proxy.actor_id.id}",
                        flush=True,
                    )
                    continue
                else:
                    # There is enough money in savings account
                    print("CheckingAccountActor.deposit", flush=True)
                    await cx_2_checking_acct_1_proxy.Deposit(send_amount)
                    print("CheckingAccountActor.withdrawal", flush=True)
                    await cx_2_checking_acct_1_proxy.Withdrawal(send_amount)
            # To deposit
            print("CustomerActor.deposit", flush=True)
            await cx_1_proxy.Deposit(send_amount)
            print("CheckingAccountActor.deposit", flush=True)
            await cx_1_checking_acct_1_proxy.Deposit(send_amount)
        else:
            print("🎲 Odd", flush=True)
            print("BankActor.transfer", flush=True)
            await bank_1_proxy.Transfer(
                {
                    "amount": send_amount,
                    "from_customer": cx_1_proxy.actor_id.id,
                    "to_customer": cx_2_proxy.actor_id.id,
                }
            )
            # From withdrawal
            print("CustomerActor.withdrawal", flush=True)
            await cx_1_proxy.Withdrawal(send_amount)
            print("CheckingAccountActor.deposit", flush=True)
            transaction_result = await cx_1_checking_acct_1_proxy.Withdrawal(
                send_amount
            )
            if transaction_result != -1:
                # Not enough money in checking account
                print("SavingsAccountActor.withdrawal", flush=True)
                transaction_result = await cx_1_savings_acct_1_proxy.Withdrawal(
                    send_amount
                )
                if transaction_result == 1:
                    # Not enough money in savings account
                    print(
                        f"Not enough money in savings account {cx_1_savings_acct_1_proxy.actor_id.id}",
                        flush=True,
                    )
                    continue
                else:
                    # There is enough money in savings account
                    print("CheckingAccountActor.deposit", flush=True)
                    await cx_1_checking_acct_1_proxy.Deposit(send_amount)
                    print("CheckingAccountActor.withdrawal", flush=True)
                    await cx_1_checking_acct_1_proxy.Withdrawal(send_amount)
            # To deposit
            print("CustomerActor.deposit", flush=True)
            await cx_2_proxy.Deposit(send_amount)
            print("CheckingAccountActor.deposit", flush=True)
            await cx_2_checking_acct_1_proxy.Deposit(send_amount)

    print("BankActor.finish", flush=True)
    rtn_obj = await bank_1_proxy.Finish()
    print(f"Return object: {rtn_obj}", flush=True)

    total = rtn_obj["end"] - rtn_obj["start"]
    print(f"Did {ITERATIONS} transactions in {total} s", flush=True)
    print(f"{ITERATIONS / total} transactions per second", flush=True)

    # Customer 1
    print("Customer 1", flush=True)
    print("CheckingAccountActor.show_balance", flush=True)
    cx_1_checking_acct_1_balance = await cx_1_checking_acct_1_proxy.ShowBalance()
    print(
        f"Checking account: {cx_1_checking_acct_1_proxy.actor_id.id}; balance: {cx_1_checking_acct_1_balance}",
        flush=True,
    )
    print("SavingsAccountActor.show_balance", flush=True)
    cx_1_savings_acct_1_balance = await cx_1_savings_acct_1_proxy.ShowBalance()
    print(
        f"Savings account: {cx_1_savings_acct_1_proxy.actor_id.id}; balance: {cx_1_savings_acct_1_balance}",
        flush=True,
    )
    # Customer 2
    print("Customer 2", flush=True)
    print("CheckingAccountActor.show_balance", flush=True)
    cx_2_checking_acct_1_balance = await cx_2_checking_acct_1_proxy.ShowBalance()
    print(
        f"Checking account: {cx_2_checking_acct_1_proxy.actor_id.id}; balance: {cx_2_checking_acct_1_balance}",
        flush=True,
    )
    print("SavingsAccountActor.show_balance", flush=True)
    cx_2_savings_acct_1_balance = await cx_2_savings_acct_1_proxy.ShowBalance()
    print(
        f"Savings account: {cx_2_savings_acct_1_proxy.actor_id.id}; balance: {cx_2_savings_acct_1_balance}",
        flush=True,
    )

    print("Please stop", flush=True)


asyncio.run(main())
