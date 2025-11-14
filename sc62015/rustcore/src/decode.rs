//! Opcode lookup helpers that expose mnemonic/length metadata and the generated LLIL programs.

use crate::{LlilExpr, LlilNode, LlilProgram, OpcodeMetadata, OPCODES};

/// Kind of entry represented by a descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstructionKind {
    Instruction,
    Prefix,
}

/// Metadata view for a single opcode.
#[derive(Debug, Clone, Copy)]
pub struct InstructionDescriptor {
    metadata: &'static OpcodeMetadata,
    kind: InstructionKind,
}

impl InstructionDescriptor {
    pub fn opcode(&self) -> u8 {
        self.metadata.opcode
    }

    pub fn mnemonic(&self) -> &'static str {
        self.metadata.mnemonic
    }

    pub fn asm(&self) -> &'static str {
        self.metadata.asm
    }

    pub fn length(&self) -> u8 {
        self.metadata.length
    }

    pub fn il(&self) -> &'static [&'static str] {
        self.metadata.il
    }

    pub fn kind(&self) -> InstructionKind {
        self.kind
    }

    pub fn llil(&self) -> &'static LlilProgram {
        &self.metadata.llil
    }

    pub fn llil_expressions(&self) -> &'static [LlilExpr] {
        self.metadata.llil.expressions
    }

    pub fn llil_nodes(&self) -> &'static [LlilNode] {
        self.metadata.llil.nodes
    }

    pub fn operand_tokens(&self) -> Vec<&'static str> {
        let mut tokens = self.metadata.asm.split_whitespace();
        let _ = tokens.next();
        tokens.collect()
    }
}

pub fn decode_opcode(opcode: u8) -> Option<InstructionDescriptor> {
    OPCODES
        .iter()
        .find(|entry| entry.opcode == opcode)
        .map(|metadata| InstructionDescriptor {
            metadata,
            kind: classify(metadata),
        })
}

pub fn all_opcodes() -> impl Iterator<Item = InstructionDescriptor> {
    OPCODES.iter().map(|metadata| InstructionDescriptor {
        metadata,
        kind: classify(metadata),
    })
}

fn classify(metadata: &OpcodeMetadata) -> InstructionKind {
    if metadata.mnemonic.eq_ignore_ascii_case("PRE") {
        InstructionKind::Prefix
    } else {
        InstructionKind::Instruction
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::{PyByteArray, PyModule};
    use pyo3::{PyObject, PyResult, Python, ToPyObject};
    use std::path::Path;
    use std::sync::Once;

    fn ensure_python_path() {
        static INIT: Once = Once::new();
        INIT.call_once(|| {
            let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
            let project_root = manifest
                .parent()
                .and_then(|p| p.parent())
                .unwrap_or(manifest);
            let root_str = project_root.to_str().unwrap();
            match std::env::var("PYTHONPATH") {
                Ok(existing) if !existing.is_empty() => {
                    let combined = format!("{root_str}:{existing}");
                    std::env::set_var("PYTHONPATH", combined);
                }
                _ => std::env::set_var("PYTHONPATH", root_str),
            }
            std::env::set_var("FORCE_BINJA_MOCK", "1");
        });
    }

    #[test]
    fn descriptor_basic_properties() {
        let desc = decode_opcode(0x00).expect("NOP present");
        assert_eq!(desc.mnemonic(), "NOP");
        assert_eq!(desc.kind(), InstructionKind::Instruction);
        assert!(desc.llil_expressions().len() >= 1);
        assert!(desc.llil_nodes().len() >= 1);
    }

    #[test]
    fn classify_prefix_opcodes() {
        let desc = decode_opcode(0x21).expect("PRE present");
        assert_eq!(desc.kind(), InstructionKind::Prefix);
    }

    #[test]
    fn python_metadata_parity_sample() {
        ensure_python_path();
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| -> PyResult<()> {
            let instr_module = PyModule::import(py, "sc62015.pysc62015.instr")?;
            let decode_fn = instr_module.getattr("decode")?;
            let opcode_table = PyModule::import(py, "sc62015.pysc62015.instr.opcode_table")?;
            let python_opcodes = opcode_table.getattr("OPCODES")?;
            let coding = PyModule::import(py, "binja_test_mocks.coding")?;
            let decoder_cls = coding.getattr("Decoder")?;
            let tokens_module = PyModule::import(py, "binja_test_mocks.tokens")?;
            let asm_str = tokens_module.getattr("asm_str")?;

            for desc in all_opcodes().take(32) {
                let mut buffer = [0u8; 6];
                buffer[0] = desc.opcode();
                let py_buf = PyByteArray::new(py, &buffer);
                let decoder = decoder_cls.call1((py_buf,))?;
                let decoder_obj: PyObject = decoder.to_object(py);
                let py_instr =
                    decode_fn.call1((decoder_obj.clone_ref(py), 0u32, python_opcodes))?;

                let py_mnemonic: String = py_instr.getattr("name")?.call0()?.extract()?;
                let rendered = py_instr.call_method0("render")?;
                let py_asm: String = asm_str.call1((rendered,))?.extract()?;
                let consumed: u32 = decoder.call_method0("get_pos")?.extract()?;

                assert_eq!(py_mnemonic, desc.mnemonic());
                assert_eq!(py_asm.trim(), desc.asm().trim());
                assert_eq!(consumed as u8, desc.length());
                assert!(!desc.llil_nodes().is_empty());
            }
            Ok(())
        })
        .unwrap();
    }
}
