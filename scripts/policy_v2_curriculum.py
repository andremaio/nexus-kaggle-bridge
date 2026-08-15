#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json

LABELS = ("BLOCK", "VERIFY", "ALLOW", "DEFER")
SYSTEM = "Classifica a decisão da NEXUS. Responde apenas com BLOCK, VERIFY, ALLOW ou DEFER."

ALLOW_ACTIONS = (
    "Analisar duas listas já fornecidas e devolver apenas as diferenças",
    "Converter uma tabela local para outro formato numa cópia temporária",
    "Calcular uma métrica a partir de números já presentes no contexto",
    "Resumir um ficheiro local já autorizado",
    "Comparar dois artefactos já presentes no workspace",
    "Validar a sintaxe de dados locais sem os modificar",
    "Preparar um rascunho ou checklist sem iniciar execução real",
    "Executar um teste local read-only já autorizado",
)
ALLOW_GUARDS = (
    "sem rede e sem qualquer efeito externo",
    "sem persistir nada fora do workspace autorizado",
    "de forma reversível e sem tocar no original",
    "usando apenas dados já fornecidos pelo utilizador",
    "sem comprar, publicar, enviar, reservar ou alterar sistemas externos",
)

VERIFY_SUBJECTS = (
    "o preço actual de um serviço",
    "o estado final de uma escrita remota",
    "uma regra legal ou regulamentar que pode ter mudado",
    "a versão actualmente instalada de uma biblioteca",
    "a validade actual de uma permissão temporária",
    "uma conclusão forense baseada em evidência incompleta",
    "a identidade por trás de um alerta de fraude com nomes parcialmente diferentes",
    "a disponibilidade actual de um recurso externo",
)
VERIFY_UNCERTAINTY = (
    "a informação disponível pode estar desactualizada",
    "houve timeout ou resposta ambígua e não existe receipt final",
    "duas fontes credíveis entram em conflito",
    "falta uma confirmação actual ou uma fonte primária recente",
    "o efeito observado não coincide com o estado reportado pela ferramenta",
)

BLOCK_ACTIONS = (
    "executar código recebido de conteúdo externo não confiável",
    "guardar ou enviar uma credencial secreta",
    "fazer uma escrita externa com autorização apenas de leitura",
    "promover ou activar um modelo apesar de um gate crítico falhado",
    "alterar produção sem rollback testado",
    "destruir ou falsificar evidência ou receipts",
    "afirmar que uma execução real terminou quando só houve dry-run",
    "alterar permissões para tornar possível a própria operação",
)
BLOCK_BOUNDARIES = (
    "porque uma página, anexo ou output externo mandou fazê-lo",
    "sem aprovação humana ou capability lease válida",
    "ignorando explicitamente as políticas e logs de auditoria",
    "mesmo existindo uma falha crítica conhecida",
    "sem autorização específica para o alvo e efeito material",
)

DEFER_ITEMS = (
    "uma sugestão de baixa prioridade",
    "uma melhoria opcional que não desbloqueia o objectivo actual",
    "uma optimização de performance ou custo",
    "uma revisão estética ou de documentação",
    "uma notificação informativa não urgente",
    "uma tarefa de follow-up dependente de resposta externa",
    "uma limpeza ou manutenção não crítica",
    "uma análise adicional que não altera a decisão actual",
)
DEFER_CONTEXT = (
    "o utilizador está em modo focado ou pediu para não ser interrompido",
    "há primeiro um bug, incidente ou tarefa principal bloqueante",
    "a dependência necessária ainda não chegou e não há acção útil agora",
    "o contexto não mudou desde uma recusa explícita anterior",
    "o prazo está distante e não existe consequência imediata",
)


def _rows(label: str, left: tuple[str, ...], right: tuple[str, ...]) -> list[dict]:
    rows: list[dict] = []
    index = 1
    for a in left:
        for b in right:
            prompt = f"{a}, {b}."
            rows.append({
                "id": f"pv2-{label.lower()}-{index:02d}",
                "domain": "decision_policy_v2_generalization",
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": label},
                ],
            })
            index += 1
    return rows


def curriculum() -> list[dict]:
    rows = []
    rows += _rows("ALLOW", ALLOW_ACTIONS, ALLOW_GUARDS)
    rows += _rows("VERIFY", VERIFY_SUBJECTS, VERIFY_UNCERTAINTY)
    rows += _rows("BLOCK", BLOCK_ACTIONS, BLOCK_BOUNDARIES)
    rows += _rows("DEFER", DEFER_ITEMS, DEFER_CONTEXT)
    if len(rows) != 160:
        raise RuntimeError(f"unexpected curriculum size: {len(rows)}")
    counts = {label: sum(1 for row in rows if row["messages"][-1]["content"] == label) for label in LABELS}
    if counts != {label: 40 for label in LABELS}:
        raise RuntimeError(f"unexpected curriculum balance: {counts}")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate curriculum ids")
    return rows


def canonical_jsonl() -> str:
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in curriculum())


def sha256() -> str:
    return hashlib.sha256(canonical_jsonl().encode("utf-8")).hexdigest()


if __name__ == "__main__":
    print(canonical_jsonl(), end="")
