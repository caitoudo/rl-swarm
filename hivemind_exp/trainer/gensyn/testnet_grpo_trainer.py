from typing import Sequence, Optional
import datasets
from hivemind_exp.chain_utils import SwarmCoordinator
from hivemind_exp.trainer.hivemind_grpo_trainer import HivemindGRPOTrainer


class TestnetGRPOTrainer(HivemindGRPOTrainer):
    def train_stage_and_save(self, trainer, train_dataset):
        super().train_stage_and_save(trainer, train_dataset)
        self.coordinator.submit_reward(
            self.node.round_num,
            self.node.stage_num,
            max(0, int(trainer.stage_rewards)),
            self.node.key,
        )

    def __init__(self, coordinator: SwarmCoordinator, **kwargs) -> None:
        self.coordinator = coordinator
        super().__init__(**kwargs)

    def submit_winners(self, round_num: int, winners: Sequence[str]):
        self.logger.info(f"🏆 Submitting winners for round {round_num}: {winners}")
        self.coordinator.submit_winners(round_num, winners[:1], self.node.key)

    def get_round_and_stage(self):
        return self.coordinator.get_round_and_stage()

    def train_stages(self, round_num, start_stage, is_coordinator):
        super().train_stages(round_num, start_stage, is_coordinator)
        self.submit_winners(round_num, self.stage_data.round_winner_fn())

    def catch_up_train(self, start_round=0, end_round=None):
        """
        补跑从 start_round 到 end_round 的轮次。
        如果 end_round 为 None，则补跑至当前轮次的前一轮。
        """
        curr_round, _ = self.get_round_and_stage()
        if end_round is None:
            end_round = curr_round - 1

        self.logger.info(f"🚀 Starting catch-up training from round {start_round} to {end_round}")
        done_rounds = set()

        for round_num in range(start_round, end_round + 1):
            if round_num in done_rounds:
                self.logger.info(f"✅ Round {round_num} already completed, skipping")
                continue
            self.logger.info(f"🔄 Catch-up training round {round_num} from stage 0")
            try:
                self.train_stages(round_num, 0, is_coordinator=False)
                done_rounds.add(round_num)
                self.cleanup()
            except datasets.exceptions.DatasetGenerationError as e:
                self.logger.error(f"❌ Dataset generation failed for round {round_num}: {e}")
                continue
            except Exception as e:
                self.logger.error(f"❌ Training failed for round {round_num}: {e}")
            break
        self.logger.info(f"🎉 Catch-up training completed from round {start_round} to {end_round}")

    def _train(self):
        try:
            curr_round, _ = self.get_round_and_stage()
            if curr_round > 0:
                self.catch_up_train(start_round=0, end_round=curr_round - 1)
                self.follower_train()
            else:
                self.follower_train()
        except Exception:
            import traceback
            traceback.print_exc()
