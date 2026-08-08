import torch
import torch.nn.functional as F

from taming.modules.vqvae.quantize import VectorQuantizer2


def _normalized_quantizer():
    quantizer = VectorQuantizer2(
        n_e=3,
        e_dim=2,
        beta=0.25,
        legacy=False,
        l2_normalized=True,
        l2_normalize_eps=1.0e-6,
    )
    with torch.no_grad():
        quantizer.embedding.weight.copy_(
            torch.tensor([[10.0, 0.0], [0.0, 3.0], [-4.0, 0.0]])
        )
    return quantizer


def test_effective_codebook_is_normalized_without_mutating_raw_weights():
    quantizer = _normalized_quantizer()
    raw_before = quantizer.embedding.weight.detach().clone()

    effective = quantizer.get_codebook_embeddings()

    torch.testing.assert_close(effective.norm(dim=-1), torch.ones(3))
    torch.testing.assert_close(quantizer.embedding.weight, raw_before)
    assert not torch.allclose(raw_before.norm(dim=-1), torch.ones(3))


def test_forward_normalizes_latents_and_effective_codes_with_gradients():
    quantizer = _normalized_quantizer()
    latent = torch.tensor([[[[3.0]], [[4.0]]]], requires_grad=True)

    quantized, loss, (_, _, indices) = quantizer(latent)

    assert indices.item() == 1
    torch.testing.assert_close(quantized.flatten(), torch.tensor([0.0, 1.0]))
    loss.backward()
    assert torch.isfinite(latent.grad).all()
    assert torch.isfinite(quantizer.embedding.weight.grad).all()


def test_all_lookup_paths_use_effective_codebook():
    quantizer = _normalized_quantizer()
    indices = torch.tensor([0, 1, 2])

    entries = quantizer.get_codebook_entry(indices, shape=None)
    decoded = quantizer.embed_code(indices.reshape(1, 1, 3))

    torch.testing.assert_close(entries.norm(dim=-1), torch.ones(3))
    torch.testing.assert_close(
        decoded.permute(0, 2, 3, 1).norm(dim=-1),
        torch.ones(1, 1, 3),
    )


def test_soft_assignment_uses_normalized_geometry_and_keeps_old_alias():
    quantizer = _normalized_quantizer()
    latent = torch.tensor([[[[3.0]], [[4.0]]]])

    probabilities = quantizer.gumbel_softmax(latent)
    normalized_latent = F.normalize(torch.tensor([[3.0, 4.0]]), dim=-1)
    normalized_codes = F.normalize(quantizer.embedding.weight, dim=-1)
    expected = F.softmax(-torch.cdist(normalized_latent, normalized_codes).square(), dim=-1)

    torch.testing.assert_close(probabilities.squeeze(0), expected)
    torch.testing.assert_close(quantizer.gumble_softmax(latent), probabilities)


def test_normalization_can_be_disabled():
    quantizer = VectorQuantizer2(
        n_e=2, e_dim=2, beta=0.25, l2_normalized=False
    )
    with torch.no_grad():
        quantizer.embedding.weight.copy_(torch.tensor([[2.0, 0.0], [0.0, 3.0]]))

    torch.testing.assert_close(
        quantizer.get_codebook_embeddings(), quantizer.embedding.weight
    )
